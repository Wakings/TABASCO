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
# get dataset
train_data_meta, train_data, test_dataset = build_dataset(args)


# make imbalance dataset 
imbalanced_train_dataset = get_imbalance_dataset(args,train_data)
# get noisy dataset 
noisy_train_dataset, noisy_transaction_matrix_real = get_noisy_dataset(train_data, args)
# make imbalance and noisy dataset 
imbalanced_and_noisy_train_dataset, noisy_transaction_matrix_real = get_noisy_dataset(imbalanced_train_dataset, args)
# # imbalanced_and_noisy_train_dataset.update()
imbalanced_and_noisy_train_loader = torch.utils.data.DataLoader(
imbalanced_and_noisy_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)




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



	feats_all, labels_all = [], []

	# Calculate initial centroids only on training data.
	with torch.no_grad():
		for idxs,(inputs, labels)  in enumerate(dataloader_):

			inputs = set_tensor(inputs,False,device)
			labels = set_tensor(labels,False,device)

			# Calculate Features of each training data
			features = forward_2(model,inputs)

			feats_all.append(features.cpu().numpy())
			labels_all.append(labels.cpu().numpy())
	
	feats = np.concatenate(feats_all)
	labels = np.concatenate(labels_all)

	featmean = feats.mean(axis=0)

	def get_centroids(feats_, labels_):
		centroids = []        
		for i in np.unique(labels_):
			centroids.append(np.mean(feats_[labels_==i], axis=0))
		return np.stack(centroids)
	# Get unnormalized centorids
	un_centers = get_centroids(feats, labels)

	# Get l2n centorids
	l2n_feats = torch.Tensor(feats.copy())
	norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
	l2n_feats = l2n_feats / norm_l2n
	l2n_centers = get_centroids(l2n_feats.numpy(), labels)

	# Get cl2n centorids
	cl2n_feats = torch.Tensor(feats.copy())
	cl2n_feats = cl2n_feats - torch.Tensor(featmean)
	norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
	cl2n_feats = cl2n_feats / norm_cl2n
	cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

	return {'mean': featmean,
			'uncs': un_centers,
			'l2ncs': l2n_centers,   
			'cl2ncs': cl2n_centers}




def bi_dimensional_sample_selection_2(args,model,loader,epoch,devices):

    model.eval()
    
    select_sample_index_list = []
    select_sample_prob_list = []
    
    select_sample_index_list_1 = []
    select_sample_prob_list_1 = []
    select_sample_prototype_1 = []
    cluster_num_list_1 = [] 

    select_sample_index_list_2 = []
    select_sample_prob_list_2 = []
    select_sample_prototype_2 = []
    cluster_num_list_2 = [] 

    select_sample_dimensionnal_status = []
    
    avg_pred_list = get_avg_pred_list(args,loader,model,devices)

    avg_pred_list_2 = get_avg_pred_list_2(args,loader,avg_pred_list,model,devices)
    cfeat = get_knn_center(model,imbalanced_and_noisy_train_loader,devices)
    mean_feat = torch.tensor(cfeat['mean']).to(devices)

    centriod_list,sample_num_list = get_centriod_list(args,avg_pred_list_2,mean_feat,model,devices)
    centriod_distance = torch.softmax(torch.einsum('ij,jk->ik',centriod_list,centriod_list.T),dim=1)

    for class_num in range(args.num_classes):

        class_dataloader = loader.run(mode = 'single',class_num = class_num)

        wjsd_info,index_info = get_wjsd_info(class_dataloader,avg_pred_list[class_num],model,devices) 
  
        acd_info, index_info = get_adaptive_centriod_distance_info(class_dataloader,centriod_list[class_num],model,mean_feat,devices) # model

        
        wjsd_info = np.array(wjsd_info)
        wjsd_info = (wjsd_info-wjsd_info.min())/(wjsd_info.max()-wjsd_info.min())
        acd_info = np.array(acd_info)
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

        if cluster_select_index_1_acd.size == 0 :
            cluster_select_index_1_acd = cluster_select_index_2_acd 
        if cluster_select_index_2_acd.size  == 0:
            cluster_select_index_2_acd = cluster_select_index_1_acd 

        acd_wjsd_mean_1 = wjsd_info[cluster_select_index_1_acd].mean(0)
        acd_wjsd_std_1 = wjsd_info[cluster_select_index_1_acd].std(0)
        acd_wjsd_pred_1 = gmm_wjsd.predict(acd_wjsd_mean_1.reshape(1, -1))[0]
        acd_wjsd_mean_2 = wjsd_info[cluster_select_index_2_acd].mean(0)
        acd_wjsd_std_2 = wjsd_info[cluster_select_index_2_acd].std(0)
        acd_wjsd_pred_2 = gmm_wjsd.predict(acd_wjsd_mean_2.reshape(1, -1))[0]

        std_list = [1] * 2
        if acd_wjsd_std_1:
            std_list[acd_wjsd_pred_1] = acd_wjsd_std_1
        else:
            std_list[acd_wjsd_pred_1] = 0.1 
        if acd_wjsd_std_2:
            std_list[acd_wjsd_pred_2] = acd_wjsd_std_2
        else:
            std_list[acd_wjsd_pred_2] = 0.1 

        if (acd_wjsd_pred_1 == acd_wjsd_pred_2 and acd_wjsd_pred_1 != gmm_wjsd.means_.argmin()) or (std_list[gmm_wjsd.means_.argmax()] / std_list[gmm_wjsd.means_.argmin()] < 0.65 ):
            select_sample_dimensionnal_status.append(1)
            select_sample_index_list_1.append(cluster_index_1_wjsd)
            select_sample_prob_list_1.append(prob_wjsd[:,gmm_wjsd.means_.argmin()][cluster_select_index_1_wjsd])

            select_sample_index_list_2.append(cluster_index_2_wjsd)
            select_sample_prob_list_2.append(prob_wjsd[:,gmm_wjsd.means_.argmax()][cluster_select_index_2_wjsd])

        else:
            select_sample_dimensionnal_status.append(0)         
            select_sample_index_list_1.append(cluster_index_1_acd)
            select_sample_prob_list_1.append(prob_acd[:,gmm_acd.means_.argmin()][cluster_select_index_1_acd])

            select_sample_index_list_2.append(cluster_index_2_acd)
            select_sample_prob_list_2.append(prob_acd[:,gmm_acd.means_.argmax()][cluster_select_index_2_acd])

    for class_num in range(args.num_classes):

        if select_sample_dimensionnal_status[class_num]:
            select_sample_index_list.extend(select_sample_index_list_1[class_num])
            select_sample_prob_list.extend(select_sample_prob_list_1[class_num])

        else:
            centriod_distance_copy = copy.deepcopy(centriod_distance[class_num])
            current_centriod_distance =copy.deepcopy(centriod_distance_copy[class_num]) 
            centriod_distance_copy[class_num] = 0
            max_centriod_distance,max_centriod_indice = centriod_distance_copy.topk(k=1, largest=True)
 
            if (abs((current_centriod_distance-max_centriod_distance[0]).item()) < 0.1 * current_centriod_distance.item()  and (sample_num_list[class_num] < sample_num_list[max_centriod_indice[0].item()])): 

                select_sample_index_list.extend(select_sample_index_list_1[class_num])
                select_sample_prob_list.extend(select_sample_prob_list_1[class_num])

            else:
   
                select_sample_index_list.extend(select_sample_index_list_2[class_num])
                select_sample_prob_list.extend(select_sample_prob_list_2[class_num])
    return select_sample_index_list,select_sample_prob_list




def get_normalization_info(info_1,info_2):
	info = np.array(info_1 + info_2)
	normal_info = (info-info.min())/(info.max()-info.min())
	normal_info_1 = normal_info[:len(info_1)].tolist()
	normal_info_2 = normal_info[len(info_1):].tolist()
	return normal_info_1,normal_info_2

def get_wjsd_info(data_loader,avg_pred,model,device):
    model.eval()
    jsd_info=[]
    index_info = [] 
    JS_dist = Jensen_Shannon()
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)

        y_f = model(input_var)
        out = torch.softmax(y_f,dim=1)

        idx = torch.tensor([x for x in range(len(out))])
        weight = out[idx,torch.argmax(out,dim=1)] / out[:,target_var[0]]
        weight = torch.clamp(weight,min=1,max = (avg_pred[torch.argmax(avg_pred,dim=0)] / avg_pred[target_var[0]]).item())

        jsd =  weight * JS_dist(out,  F.one_hot(target_var, num_classes = args.num_classes))

        jsd_info.extend(jsd.tolist())
        index_info.extend(indexs.tolist())
    return jsd_info,np.array(index_info)

def get_jsd_info(data_loader,avg_pred,model,class_num,device):
    model.eval()
    jsd_info=[]
    index_info = [] 
    JS_dist = Jensen_Shannon()

    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)
        target_var = set_tensor(target, False, device)

        y_f = model(input_var)
        out = torch.softmax(y_f,dim=1)
        

        idx = torch.tensor([x for x in range(len(out))])
        weight = out[idx,torch.argmax(out,dim=1)] / out[:,target_var[0]]

        weight = torch.clamp(weight,min=1,max = (avg_pred[torch.argmax(avg_pred,dim=0)] / avg_pred[target_var[0]]).item())

        jsd =  JS_dist(out,  F.one_hot(target_var, num_classes = args.num_classes))

        jsd_info.extend(jsd.tolist())
        index_info.extend(indexs.tolist())
    return jsd_info,np.array(index_info)

def get_adaptive_centriod_distance_info(data_loader,centriod,model,meat_feat,devices):
    model.eval()
    dist_info = []
    index_info = []
    for i, (input, target,indexs) in enumerate(data_loader):


        input_var = set_tensor(input, False, devices)

        features = forward_2(model,input_var) - meat_feat
        features = F.normalize(features,p=2,dim=1)

        dist = torch.einsum('ij,j->i',features,centriod.T)
        dist_info.extend(dist.tolist())
        index_info.extend(indexs.tolist())
    return dist_info,np.array(index_info)


def get_adaptive_centriod_distance_info_(data_loader,centriod,model,meat_feat,devices):
    model.eval()
    dist_info = []
    index_info = []
    feature_info = []
    for i, (input, target,indexs) in enumerate(data_loader):


        input_var = set_tensor(input, False, devices)

        features_ = forward_2(model,input_var) - meat_feat
        features = F.normalize(features_,p=2,dim=1)

        dist = torch.einsum('ij,j->i',features,centriod.T)
        dist_info.extend(dist.tolist())
        index_info.extend(indexs.tolist())
        feature_info.extend(features_.tolist())
    return dist_info,np.array(index_info),np.array(feature_info)
        
def get_adaptive_centriod(args,global_dataloader,local_dataloader,avg_pred,class_num,feat_mean,model,devices):

    high_confidence_samples,sample_num = get_high_confidence_samples(local_dataloader,avg_pred,class_num,model,devices)

    adptive_feat_c = high_confidence_samples - feat_mean
    adptive_feat_cl2 = F.normalize(adptive_feat_c,p=2,dim=1)
    adptive_centriod  = adptive_feat_cl2.mean(0)
    return adptive_centriod,sample_num

def get_high_confidence_samples(global_dataloader,avg_pred,class_num,model,devices):
    select_features_list = torch.tensor([]).to(devices)
    avg_pred = avg_pred[class_num]
    sample_num = 0
    for i,(input, target,indexs) in enumerate(global_dataloader):

        input_var = set_tensor(input, False, device)

        features = forward_2(model,input_var)
        y_f = model.linear(features)
        preds = torch.softmax(y_f,dim=1)
        arg_idx = torch.argmax(preds,dim=1)
        select_ = torch.eq(arg_idx,torch.argmax(avg_pred))
        get_high_confidence_criterion = avg_pred[torch.argmax(avg_pred)]
        select_index = torch.gt(preds[:,torch.argmax(avg_pred)],get_high_confidence_criterion)

        select_features = features[select_index*select_]
        sample_num += (select_index*select_).sum().item()
        select_features_list = torch.cat([select_features_list,select_features],dim=0)

    if sample_num == 0 :
        for i,(input, target,indexs) in enumerate(global_dataloader):

            input_var = set_tensor(input, False, device)

            features = forward_2(model,input_var)

            select_features = features
            sample_num += len(target)
            select_features_list = torch.cat([select_features_list,select_features],dim=0)
        
    return select_features_list,sample_num

def get_avg_pred(data_loader,model,devices):
    model.eval()

    avg_pred = torch.tensor([]).to(devices)
    for i, (input, target,indexs) in enumerate(data_loader):

        input_var = set_tensor(input, False, device)

        y_f = model(input_var)
        out = torch.softmax(y_f,dim=1).mean(0).unsqueeze(0)
        avg_pred = torch.cat([avg_pred,out],dim=0)

    return avg_pred.mean(0)

def get_avg_pred_2(data_loader,avg_pred_2,model,device):
	model.eval()
	avg_pred = torch.tensor([]).to(device)
	avg_argmax = torch.argmax(avg_pred_2,dim=0)
	for i, (input, target,indexs) in enumerate(data_loader):

		input_var = set_tensor(input, False, device)


		y_f = model(input_var)
		out = torch.softmax(y_f,dim=1)
		idx = [i for i in range(target.shape[0])]
		weight = torch.clamp(out[idx,avg_argmax] / avg_pred_2[avg_argmax],min=1)
		out[idx,avg_argmax] = weight * out[idx,avg_argmax]
		avg_pred = torch.cat([avg_pred,out.mean(0).unsqueeze(0)],dim=0)


	return avg_pred.mean(0)
def get_avg_pred_list(args,loader,model,devices):

    avg_pred_list = torch.tensor([]).to(devices)
    for class_num in range(args.num_classes):

        class_dataloader = loader.run(mode='single',class_num=class_num)
        avg_pred = get_avg_pred(class_dataloader,model,devices).unsqueeze(0) 

        avg_pred_list = torch.cat([avg_pred_list,avg_pred],dim=0)
        del class_dataloader
    return avg_pred_list

def get_avg_pred_list_2(args,loader,avg_pred_list,model,devices):

    avg_pred_list_2 = torch.tensor([]).to(devices)
    for class_num in range(args.num_classes):

        class_dataloader = loader.run(mode='single',class_num=class_num)
        avg_pred = get_avg_pred_2(class_dataloader,avg_pred_list[class_num],model,devices).unsqueeze(0) 

        avg_pred_list_2 = torch.cat([avg_pred_list_2,avg_pred],dim=0)
        del class_dataloader
    return avg_pred_list_2

def get_centriod_list(args,avg_pred_list_2,mean_feat,model,devices):
    centriod_list = torch.tensor([]).to(devices)
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
loader = cifar_dataloader(args.dataset,r=args.corruption_prob,imb_factor=args.imb_factor,noise_mode=args.corruption_type,batch_size=args.batch_size,num_workers=5,\
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

def main(warm_up_flag):

    if not warm_up_flag:
        net1.load_state_dict(torch.load(save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net1.pt'))
        net2.load_state_dict(torch.load(save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net2.pt'))
     
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

if __name__ == '__main__':
    main(warm_up_flag)
    