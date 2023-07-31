from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import copy
import os 
import shutil
import re
class red_mini_imagenet_dataset(Dataset): 
    def __init__(self, root, train_imgs,train_labels,val_labels,transform, mode, val_imgs=[],select_num=0, select_index=[], probability=[],  num_class=100): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = train_labels
        self.val_labels = val_labels       
        self.probability = probability
        self.train_imgs = train_imgs

        if mode == 'all':

            self.train_imgs = train_imgs
        elif self.mode == "labeled":   

            self.train_imgs = [i for i in select_index]                

            print("select num :",len(select_index),end=" ")

        elif self.mode == "unlabeled":   

            select_path = list(set(self.train_imgs).difference(set(select_index))) 
            self.train_imgs = [i for i in select_path]                
         
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs))) 
        elif self.mode == "select":
            self.train_imgs = [i for i in select_index]   
                                                        
        elif mode=='val':
            self.val_imgs = val_imgs

        else:
            data_list = [i for i in train_imgs if self.train_labels[i] == select_num]
            if self.mode == "single" :
                self.train_imgs = [i for i in data_list] 
     
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path        
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target          
        else:
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path 
            
        
    def __len__(self):
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            
        
class red_mini_imagenet_dataloader():  
    def __init__(self, root,pre_treatment, batch_size, num_batches, num_workers, num_class,imb_factor=0.1,noise_ratio=0.4):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
        self.clean_labels = {}
        self.noise_labels = {}
        self.train_labels = {}
        self.val_labels = {}   
        self.num_class = num_class
        self.train_imgs = []
        self.noisy_imgs = []
        self.val_imgs = [] 
        self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(84),
                transforms.ColorJitter(brightness=0.4,contrast = 0.4 ,saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),                   
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize([int(84*1.15),int(84*1.15)]),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])     

        
        if pre_treatment:
            print("pre_treatment:")  
            if os.path.exists(self.root + '/validation'):  

                for root, dirs, files in os.walk(self.root + '/validation'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        shutil.copy(file_path, self.root + '/validation')
        control_label_path = self.root + '/split'
        with open('%s/blue_noise_nl_0.0'%control_label_path,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = self.root + '/all_images'+ '/' +entry[0]
                self.train_imgs.append(img_path)
                self.clean_labels[img_path] = int(entry[1]) 

        with open('%s/red_noise_nl_%.1f'%(control_label_path,0.8),'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                if  re.match('^n.*',entry[0])  is None:   
                    img_path = self.root + '/all_images'+ '/' +entry[0]
                    self.noisy_imgs.append(img_path)
                    self.noise_labels[img_path] = int(entry[1])
        random.shuffle(self.noisy_imgs)

        with open('%s/clean_validation'%control_label_path,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split() 
                     
                img_path = self.root + '/validation'+ '/' +entry[0]
                
                self.val_imgs.append(img_path)
                self.val_labels[img_path] = int(entry[1]) 


        img_num_list = get_img_num_per_cls(len(self.train_imgs) / num_class, num_class, imb_factor, 0)

        self.train_imgs =sample_dataset(self.train_imgs ,self.clean_labels,img_num_list,num_class,'select')
        self.data_num = sum(img_num_list)
        select_noisy_num = int(self.data_num / (1 - noise_ratio) - self.data_num)

        self.train_imgs.extend(self.noisy_imgs[:select_noisy_num])
        self.train_labels.update(self.clean_labels)
        self.train_labels.update(self.noise_labels)
        self.data_num = len(self.train_imgs)
        cal_label_distribution(self.train_imgs,self.train_labels)
    def run(self,mode,select_num=0,select_index=[],prob=[]):        
        if mode=='warmup':
            warmup_dataset = red_mini_imagenet_dataset(self.root,self.train_imgs,self.train_labels,self.val_labels,transform=self.transform_train, mode='all')
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = red_mini_imagenet_dataset(self.root,self.train_imgs,self.train_labels,self.val_labels,transform=self.transform_train, mode='labeled',select_index=select_index, probability=prob)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = red_mini_imagenet_dataset(self.root,self.train_imgs,self.train_labels,self.val_labels,transform=self.transform_train, mode='unlabeled',select_index=select_index, probability=prob)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = red_mini_imagenet_dataset(self.root,self.train_imgs,self.train_labels,self.val_labels,transform=self.transform_test, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader                      
        elif mode=='val':
            val_dataset = red_mini_imagenet_dataset(self.root,self.train_imgs,self.train_labels,self.val_labels,transform=self.transform_test, mode='val',val_imgs=self.val_imgs)
            val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=50,
                shuffle=False,
                num_workers=self.num_workers)             
            return val_loader              
        else:
            eval_dataset = red_mini_imagenet_dataset(self.root,self.train_imgs,self.train_labels,self.val_labels,transform=self.transform_test, mode=mode,select_num=select_num,select_index=select_index, probability=prob)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader       

def cal_label_distribution(train_data,train_label):
    class_num_list = []
    data_list= {}
    for j in range(100):
        data_list[j] = [i for i in train_data if train_label[i] == j]
        class_num_list.append(len(data_list[j]))
    print(class_num_list)
def get_imbalance_ratios(imb_factor, cls_num):
    imbalance_ratios = []
    for cls_idx in range(cls_num):
        ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
        imbalance_ratios.append(ratio)
    return imbalance_ratios

def get_img_num_per_cls(img_num,cls_num,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    img_max = img_num
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    imbalance_ratios = get_imbalance_ratios(imb_factor, cls_num)
    for cls_idx in range(cls_num):
        ratio = imbalance_ratios[cls_idx]
        num = img_max * ratio
        img_num_per_cls.append(int(num))
    return img_num_per_cls

from operator import itemgetter
def sample_dataset(train_data,train_label, img_num_list, num_classes, kind):
    """
    Args:
        dataset
        img_num_list
        num_classes
        kind
    Returns:

    
    """
    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i in train_data if train_label[i] == j]

    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        '''
        cls_idx : class index
        img_id_list:sample global index list
        data_list:{'cls_idx':[img_id_list],}
        '''
        np.random.shuffle(img_id_list)
        # print(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        # print(img_num)
        if kind=='delete':
            idx_to_del.extend(img_id_list[:img_num])
        else:
            idx_to_del.extend(img_id_list[img_num:])
    train_data_ = list(set(train_data).difference(set(idx_to_del))) 

    return train_data_
