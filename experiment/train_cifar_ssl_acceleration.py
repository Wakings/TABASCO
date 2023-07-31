import os
import sys
from matplotlib.pyplot import get

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from utils.util import *
from settings.configs import *
from datas.data_utils import *
from models.resnet import *
from models.PreResNet import *
from validates.validation import *
from math import inf
import string
import random
from sklearn import metrics

# get shell arg
args = parse_arguments()

# fixed seed
set_seed(args.seed)
# device detect
device = set_device(args)
# load config
config = load_config()

# Dataloader set
kwargs = {'num_workers': config['Dataloader_set']['num_workers'],
          'pin_memory': config['Dataloader_set']['pin_memory']}


def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s-%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                    %(args.dataset, args.imb_factor, args.corruption_type,args.corruption_prob, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
            sys.stdout.flush()



def warmup_net(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1

    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)       
        loss.backward()  
        optimizer.step() 
        if batch_idx % args.print_freq == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s-%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                    %(args.dataset, args.imb_factor, args.corruption_type,args.corruption_prob, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()  


def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
    
    
def create_model():
    model = ResNet18(num_classes=args.num_classes)
    model = model.cuda()
    return model

def get_knn_center(model,dataloader_,device):
	print('===> Calculating KNN centroids.')

	feats_all = torch.tensor([],device=device)
	# Calculate initial centroids only on training data.
	with torch.no_grad():
		for idxs,(inputs, labels,_)  in enumerate(dataloader_):

			inputs = set_tensor(inputs,False,device)
			labels = set_tensor(labels,False,device)


			# Calculate Features of each training data
			features = forward_2(model,inputs)

			feats_all = torch.cat([feats_all,features],dim=0)


	featmean = feats_all.mean(0)


	return {'mean': featmean}





def bi_dimensional_sample_selection_2(args,model,loader,epoch,devices):

    model.eval()

    select_sample_index_list = []
    select_sample_prob_list = []
    

    warmup_trainloader = loader.run('all')
    cfeat = get_knn_center(model,warmup_trainloader,devices)
    avg_pred_list = get_avg_pred_list_3(warmup_trainloader,model,devices)

    avg_pred_list_2 = get_avg_pred_list_4(warmup_trainloader,avg_pred_list,model,devices)

    mean_feat = cfeat['mean']

    centriod_list,sample_num_list = get_adaptive_centriod_2(args,warmup_trainloader,avg_pred_list_2,mean_feat,model,devices)

    centriod_distance = torch.softmax(torch.einsum('ij,jk->ik',centriod_list,centriod_list.T),dim=1)

    wjsd_infos,index_infos,targets = get_wjsd_info_2(warmup_trainloader,avg_pred_list,model,devices) 
  
    acd_infos= get_adaptive_centriod_distance_info_2(warmup_trainloader,centriod_list,model,mean_feat,devices) 
    for class_num in range(args.num_classes):
        # bi_dimensional_sample_separation

        select_index = targets == class_num
       

        index_info = index_infos[select_index].cpu().numpy()
        wjsd_info = wjsd_infos[select_index].cpu().numpy() 
        wjsd_info = (wjsd_info-wjsd_info.min())/(wjsd_info.max()-wjsd_info.min())
        acd_info = acd_infos[select_index].cpu().numpy()
        acd_info = (acd_info-acd_info.min())/(acd_info.max()-acd_info.min())

        combine_wjsd = wjsd_info.reshape(-1,1)
        combine_acd = acd_info.reshape(-1,1)

        prob_wjsd, gmm_wjsd= gmm_fit_func(combine_wjsd)  
        prob_acd, gmm_acd= gmm_fit_func(combine_acd)        

        cluster_select_index_1_wjsd = (prob_wjsd[:,gmm_wjsd.means_.argmin()]>0.5)
        cluster_select_index_2_wjsd = ~cluster_select_index_1_wjsd
        cluster_index_1_wjsd = index_info[cluster_select_index_1_wjsd]
        cluster_index_2_wjsd = index_info[cluster_select_index_2_wjsd]
        
        cluster_select_index_1_acd = (prob_acd[:,gmm_acd.means_.argmin()]>0.5)
        cluster_select_index_2_acd = ~cluster_select_index_1_acd
        cluster_index_1_acd = index_info[cluster_select_index_1_acd]
        cluster_index_2_acd = index_info[cluster_select_index_2_acd]        
    
        acd_wjsd_mean_1 = wjsd_info[cluster_select_index_1_acd].mean(0)
        acd_wjsd_std_1 = wjsd_info[cluster_select_index_1_acd].std(0)
        acd_wjsd_pred_1 = gmm_wjsd.predict(acd_wjsd_mean_1.reshape(1, -1))[0]

        acd_wjsd_mean_2 = wjsd_info[cluster_select_index_2_acd].mean(0)
        acd_wjsd_std_2 = wjsd_info[cluster_select_index_2_acd].std(0)
        acd_wjsd_pred_2 = gmm_wjsd.predict(acd_wjsd_mean_2.reshape(1, -1))[0]

        std_list = [1] * 2
        std_list[acd_wjsd_pred_1] = acd_wjsd_std_1
        std_list[acd_wjsd_pred_2] = acd_wjsd_std_2


        if (acd_wjsd_pred_1 == acd_wjsd_pred_2 and acd_wjsd_pred_1 != gmm_wjsd.means_.argmin()) or (std_list[gmm_wjsd.means_.argmax()] / std_list[gmm_wjsd.means_.argmin()] < 0.65 ):
            select_sample_index_list.extend(cluster_index_1_wjsd)
            select_sample_prob_list.extend(prob_wjsd[:,gmm_wjsd.means_.argmin()][cluster_select_index_1_wjsd])
        else:
            centriod_distance_copy = copy.deepcopy(centriod_distance[class_num])
            current_centriod_distance =copy.deepcopy(centriod_distance_copy[class_num]) 
            centriod_distance_copy[class_num] = 0
            max_centriod_distance,max_centriod_indice = centriod_distance_copy.topk(k=1, largest=True)
            if (abs((current_centriod_distance-max_centriod_distance[0]).item()) < 0.1 * current_centriod_distance.item()  and (sample_num_list[class_num] < sample_num_list[max_centriod_indice[0].item()])): 
                select_sample_index_list.extend(cluster_index_1_acd)
                select_sample_prob_list.extend(prob_acd[:,gmm_acd.means_.argmin()][cluster_select_index_1_acd])
            else:

                select_sample_index_list.extend(cluster_index_2_acd)
                select_sample_prob_list.extend(prob_acd[:,gmm_acd.means_.argmax()][cluster_select_index_2_acd])

    return select_sample_index_list,select_sample_prob_list




def get_normalization_info(info_1,info_2):
	info = np.array(info_1 + info_2)
	normal_info = (info-info.min())/(info.max()-info.min())
	normal_info_1 = normal_info[:len(info_1)].tolist()
	normal_info_2 = normal_info[len(info_1):].tolist()
	return normal_info_1,normal_info_2

def get_wjsd_info_2(data_loader,avg_pred,model,device):
    model.eval()

    JS_dist = Jensen_Shannon()
    targets = torch.tensor([],device=device)
    jsd_info = torch.tensor([],device=device)
    index_info = torch.tensor([],dtype=torch.long)
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)
        targets = torch.cat([targets,target_var],dim=0)
        y_f = model(input_var)
        out = torch.softmax(y_f,dim=1)

        idx = torch.tensor([x for x in range(len(out))])
        weight = out[idx,torch.argmax(out,dim=1)] / out[idx,target_var]

        weight_max = (avg_pred[target_var,torch.argmax(avg_pred[target_var],dim=1)] / avg_pred[target_var,target_var]).detach()
        weight_index  = weight > weight_max
        weight[weight_index] = weight_max[weight_index]

        jsd =  weight * JS_dist(out,  F.one_hot(target_var, num_classes = args.num_classes))

        jsd_info = torch.cat([jsd_info,jsd],dim=0)

        index_info = torch.cat([index_info,indexs],dim=0)
    return jsd_info,index_info,targets

def get_adaptive_centriod_distance_info_2(data_loader,centriod,model,meat_feat,devices):
    model.eval()
    dist_info = torch.tensor([],device=devices)

    
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, devices)
        target_var = set_tensor(target, False, device) 
        features = forward_2(model,input_var) - meat_feat
        features = F.normalize(features,p=2,dim=1)

        dist = torch.einsum('ij,ji->i',features,centriod[target_var].T)

        dist_info = torch.cat([dist_info,dist],dim=0)

    return dist_info


    
def get_high_confidence_samples_2(global_dataloader,avg_pred,model,devices):
    select_features_list = torch.tensor([],device=devices)
    sample_num = 0
    targets = torch.tensor([],device=devices)
    for i,(input, target,indexs) in enumerate(global_dataloader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)
        features = forward_2(model,input_var)
        y_f = model.linear(features)
        preds = torch.softmax(y_f,dim=1)
        arg_idx = torch.argmax(preds,dim=1)
        select_ = torch.eq(arg_idx,torch.argmax(avg_pred[target_var],dim=1))

        idx = [i for i in range(target_var.shape[0])]
        get_high_confidence_criterion = avg_pred[target_var,torch.argmax(avg_pred[target_var],dim=1)]
        select_index = torch.gt(preds[idx,torch.argmax(avg_pred[target_var],dim=1)],get_high_confidence_criterion)

        
        select_features = features[select_index*select_]

        select_features_list = torch.cat([select_features_list,select_features],dim=0)
        targets = torch.cat([targets,target_var[select_index*select_]],dim=0)
    return select_features_list,targets 

def get_adaptive_centriod_2(args,local_dataloader,avg_pred,feat_mean,model,devices):

    adptive_centriod_list = torch.tensor([],device=devices)
    sample_num_list = []
    high_confidence_samples,targets = get_high_confidence_samples_2(local_dataloader,avg_pred,model,devices)

    adptive_feat_c = high_confidence_samples - feat_mean
    adptive_feat_cl2 = F.normalize(adptive_feat_c,p=2,dim=1)
    for i in range(args.num_classes):
        adptive_centriod_list = torch.cat([adptive_centriod_list,adptive_feat_cl2[targets == i].mean(0).unsqueeze(0)],dim=0)
        sample_num_list.append((targets == i).sum(0).item())

    return adptive_centriod_list,sample_num_list

def get_avg_pred_list_3(data_loader,model,devices):
    model.eval()

    preds = torch.tensor([],device=devices)
    targets = torch.tensor([],device=devices)
    avg_pred = torch.tensor([],device=devices)
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)
        y_f = model(input_var)
        out = torch.softmax(y_f,dim=1)
        preds = torch.cat([preds,out],dim=0)
        targets = torch.cat([targets,target_var],dim=0)    

    for i in range(args.num_classes):
        avg_pred = torch.cat([avg_pred,preds[targets == i].mean(0).unsqueeze(0)],dim=0)
    return avg_pred

def get_avg_pred_list_4(data_loader,avg_pred_2,model,devices):
    model.eval()
    avg_pred = torch.tensor([],device=devices)# .to(device)
    avg_argmax = torch.argmax(avg_pred_2,dim=1)
    preds = torch.tensor([],device=devices)
    targets = torch.tensor([],device=devices)
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)

        y_f = model(input_var)
        out = torch.softmax(y_f,dim=1)
        idx = [i for i in range(target.shape[0])]
        weight = torch.clamp(out[idx,avg_argmax[target_var]] / avg_pred_2[target_var,avg_argmax[target_var]],min=1)
        out[idx,avg_argmax[target_var]] = weight * out[idx,avg_argmax[target_var]]
        preds = torch.cat([preds,out],dim=0)
        targets = torch.cat([targets,target_var],dim=0)
    for i in range(args.num_classes):
        avg_pred = torch.cat([avg_pred,preds[targets == i].mean(0).unsqueeze(0)],dim=0)
    return avg_pred

def get_centriod_list(args,avg_pred_list_2,mean_feat,model,devices):
    centriod_list = torch.tensor([],device=devices)
    sample_num_list = []
    for class_num in range(args.num_classes):
        class_dataloader = loader.run(mode='single',class_num=class_num)
        centriod,sample_num = get_adaptive_centriod(args,'other_class_dataloader',class_dataloader,avg_pred_list_2,class_num,mean_feat,model,devices) 
        centriod_list = torch.cat([centriod_list,centriod.unsqueeze(0)],dim=0)
        sample_num_list.append(sample_num)
        del class_dataloader
    return centriod_list,sample_num_list
# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

def gmm_fit_func(input_loss):
    input_loss = np.array(input_loss)

    gmm = GaussianMixture(n_components=2,max_iter=30,tol=1e-2,reg_covar=5e-4) 
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 

    return prob,gmm
## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

def forward_1(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

def forward_2(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)

        return out



stats_log=open('./checkpoint/%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'_acc.txt','w')     
loader = cifar_dataloader(args.dataset,r=args.corruption_prob,imb_factor=args.imb_factor,noise_mode=args.corruption_type,batch_size=args.batch_size,num_workers=8,\
    root_dir=args.data_path,log=stats_log)
if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

print('| Building net')
net1 = create_model()
net2 = create_model()

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CEloss = nn.CrossEntropyLoss()
if args.corruption_type=='flip':
    conf_penalty = NegEntropy()
test_loader = loader.run('test')  
save_warmup_path =   './checkpoint/'
warm_up_flag = True
eval_flag = False
recover_flag = False

def main(warm_up_flag):

    if not warm_up_flag:
        net1.load_state_dict(torch.load(save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net1.pt'))
        net2.load_state_dict(torch.load(save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net2.pt'))
        print("load warmup model")
      
    for epoch in range(args.num_epochs+1):    
        lr=args.lr  
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr       
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr          

        
        if  warm_up_flag: 
            if  epoch<warm_up:     
                warmup_trainloader = loader.run('warmup')
                print('Warmup Net1')
                warmup_net(epoch,net1,optimizer1,warmup_trainloader)    
                print('\nWarmup Net2')
                warmup_net(epoch,net2,optimizer2,warmup_trainloader) 
                if epoch == warm_up -1:
                    torch.save(net1.state_dict(),save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net1.pt') # _1
                    torch.save(net2.state_dict(),save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net2.pt') # _2
                    warm_up_flag = False
        else:  
            if epoch < warm_up:
                continue 
            else:      

                with torch.no_grad():         
                    select_index_1,select_prob_1 = bi_dimensional_sample_selection_2(args,net1,loader,epoch,devices=device)    
                    
                    select_index_2,select_prob_2 = bi_dimensional_sample_selection_2(args,net2,loader,epoch,devices=device)    

                print('Train Net1')
                labeled_trainloader, unlabeled_trainloader = loader.run('train',select_index=select_index_2,prob = select_prob_2) # co-divide
                
                train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader) # train net1  
              
                print('\nTrain Net2')
                labeled_trainloader, unlabeled_trainloader = loader.run('train',select_index=select_index_1,prob =select_prob_1) # co-divide
                train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader) # train net2  

        test(epoch,net1,net2) 
        if epoch % 33 == 0:
            torch.save(net1.state_dict(),save_warmup_path+'%s_%.1f_%s_%.1f_'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+str(epoch)+'train_net1.pt') 
            torch.save(net1.state_dict(),save_warmup_path+'%s_%.1f_%s_%.1f_'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+str(epoch)+'train_net2.pt') 

if __name__ == '__main__':
    main(warm_up_flag)
    