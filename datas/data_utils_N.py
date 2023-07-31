from select import select
from matplotlib.pyplot import axis
import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import copy
import random
import numpy as np


from PIL import Image
from PIL import ImageFilter
from sklearn.preprocessing import LabelEncoder
from typing import Any, Callable, Optional, Tuple



from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import copy
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, train_data,noise_label,data_index,dataset, root_dir, transform, mode, class_num=0, select_index=[], probability=[],train_label=[],log=''): 
        

        self.transform = transform
        self.mode = mode  
        self.data_index = data_index
        self.probability = np.array(probability)
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:        
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
  
            elif self.mode == "labeled":
                clean = (np.array(noise_label[select_index])==np.array(train_label[select_index])) 

                print("select num :",len(select_index),end=" ")
                print("select clean acc:",np.sum(clean)/len(select_index))

                self.train_data = train_data[select_index]
                self.noise_label = noise_label[select_index]  
                   
            elif self.mode == "unlabeled" or self.mode == "unlabeled_":                                         
            
                self.train_data = np.delete(train_data,select_index ,axis=0)
                self.noise_label = np.delete(noise_label,select_index ,axis=0) 
            elif self.mode == 'select':
                self.train_data = train_data[select_index]
                self.noise_label = noise_label[select_index]     
            else: 
                data_list = [i for i,label in enumerate(noise_label) if label == class_num ]  
                if self.mode == 'single':
                    self.train_data = train_data[data_list]
                    self.noise_label = noise_label[data_list]  

                    self.data_index = data_index[data_list]
                elif self.mode == 'other': 
                    self.train_data = np.delete(train_data,data_list ,axis=0)
                    self.noise_label = np.delete(noise_label,data_list ,axis=0)  
                elif self.mode == 'clean':

                    clean_data_list = []
                    for idx in data_list:
                        if noise_label[idx] == train_label[idx]:
                            clean_data_list.append(idx)
                    self.train_data = train_data[clean_data_list]
                    self.noise_label = noise_label[clean_data_list] 
                elif self.mode == 'noisy':
                    noisy_data_list = copy.deepcopy(data_list)
                    for idx in data_list:
                        if noise_label[idx] == train_label[idx]:
                            noisy_data_list.remove(idx)   
                    self.train_data = train_data[noisy_data_list]
                    self.noise_label = noise_label[noisy_data_list]                  
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        else:
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, self.data_index[index]     
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        


class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3
        

class cifar_dataloader():  
    def __init__(self, dataset, r, imb_factor,noise_mode, batch_size, num_workers, root_dir,log):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])  
        train_data=[]
        train_label=[]
        human_label = []
        if dataset=='cifar10': 
            num_classes = 10
            for n in range(1,6):
                dpath = '%s/data_batch_%d'%(root_dir,n)
                data_dic = unpickle(dpath)
                train_data.append(data_dic['data'])
                train_label = train_label+data_dic['labels']
            train_data = np.concatenate(train_data)
            if noise_mode not in ['unif','flip']:
                noise_label_ = torch.load('%s/CIFAR-10_human.pt'%root_dir)
                human_label = noise_label_[noise_mode].reshape(-1) 
                print(len(human_label))
            else:
                human_label = None
        elif dataset=='cifar100':   
            num_classes = 100 
            train_dic = unpickle('%s/train'%root_dir)
            train_data = train_dic['data']
            train_label = train_dic['fine_labels']
            if noise_mode not in ['unif','flip']:
                noise_label_ = torch.load('%s/CIFAR-100_human.pt'%root_dir)
                human_label = noise_label_[noise_mode].reshape(-1) 
                print(len(human_label))
            else:
                human_label = None
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))
        # imbalance 
        img_num_list = get_img_num_per_cls_1(dataset, num_classes, imb_factor, 0)
        print("img_num_list:", img_num_list)
        print("img_num_list-sum:", sum(img_num_list))
        train_data ,train_label,human_label=sample_dataset_1(train_data ,train_label,img_num_list,num_classes,'select',human_label)
        self.data_num = sum(img_num_list)
        # noisy
        if noise_mode=='unif':
            noisy_transaction_matrix_real = uniform_mix_c_1(self.r, num_classes)
        elif noise_mode=='flip':
            noisy_transaction_matrix_real = flip_labels_c_1(self.r, num_classes)
        if noise_mode in ['unif','flip']:
            noisy_label = copy.deepcopy(train_label)
            for i in range(sum(img_num_list) ):
                noisy_label[i] = np.random.choice(num_classes, p=noisy_transaction_matrix_real[train_label[i]])
            print("noisy transation matrix:",noisy_transaction_matrix_real) 
        else:
            noisy_label = human_label
        
        self.train_data = train_data
        self.train_label = train_label
        self.noise_label = noisy_label
        self.data_index  = np.array([i for i in range(len(noisy_label))])
    def run(self,mode,class_num=0,select_index=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset,  root_dir=self.root_dir, transform=self.transform_train, mode="all")                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", select_index=select_index, probability=prob, train_label=self.train_label,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset,  root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",  select_index=select_index)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset,  root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        elif mode == 'eval_train':
            all_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset,  root_dir=self.root_dir, transform=self.transform_train, mode="all")                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        else :
            eval_dataset = cifar_dataset(self.train_data,self.noise_label,self.data_index,dataset=self.dataset, root_dir=self.root_dir, transform=self.transform_test, mode=mode,class_num=class_num,select_index = select_index,train_label=self.train_label)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        


def sample_dataset_1(train_data,train_label, img_num_list, num_classes, kind,human_label=None):
    """
    Args:
        dataset
        img_num_list
        num_classes
        kind
    Returns:

    
    """
    if human_label is None:
        human_label = copy.deepcopy(train_label)
    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(train_label) if label == j]

    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        '''
        cls_idx : class index
        img_id_list:sample global index list
        data_list:{'cls_idx':[img_id_list],}
        '''
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        if kind=='delete':
            idx_to_del.extend(img_id_list[:img_num])
        else:
            idx_to_del.extend(img_id_list[img_num:])

    train_label = np.delete(train_label, idx_to_del, axis=0)
    human_label = np.delete(human_label, idx_to_del, axis=0)
    train_data = np.delete(train_data, idx_to_del, axis=0)
    return train_data,train_label,human_label



def get_imbalance_ratios_1(imb_factor, cls_num):
    imbalance_ratios = []
    for cls_idx in range(cls_num):
        ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
        imbalance_ratios.append(ratio)
    return imbalance_ratios

def get_img_num_per_cls_1(dataset,cls_num,imb_factor=None,num_meta=None):
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
    if dataset == 'cifar10':
        img_max = 5000-num_meta

    if dataset == 'cifar100':
        img_max = 500-num_meta

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    imbalance_ratios = get_imbalance_ratios_1(imb_factor, cls_num)
    for cls_idx in range(cls_num):
        ratio = imbalance_ratios[cls_idx]
        num = img_max * ratio
        img_num_per_cls.append(int(num))
    return img_num_per_cls

def uniform_mix_c_1(mixing_ratio, num_classes):
    """
    returns a linear interpolation of a uniform matrix and an identity matrix
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_c_1(corruption_prob, num_classes, seed=25):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    torch.save(C, 'noisy_transaction_matrix_real.pt')
    return C
















def get_transform(args):
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform_train = transforms.Compose([

    transforms.Pad(padding=4, fill=0, padding_mode="reflect"),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    transform_for_contrast = TwoCropTransform(transforms.Compose([

    transforms.Pad(padding=4, fill=0, padding_mode="reflect"),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
    transforms.RandomGrayscale(p=0.2),

    transforms.ToTensor(),
    normalize
    ]))

    return transform_train,transform_test,transform_for_contrast


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



class CIFAR10_With_Index(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index), index
    def __len__(self) -> int:
        return super().__len__()

class CIFAR100_With_Index(torchvision.datasets.CIFAR100):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index), index
    def __len__(self) -> int:
        return super().__len__()
        
def build_dataset(args,transform_type = 'normal'):
    transform_train,transform_test,transform_for_contrast = get_transform(args)
    if transform_type == 'normal':
        transforms = transform_train
    else:
        transforms = transform_for_contrast

    if args.dataset == 'cifar10':
        if args.with_index:
            train_dataset = CIFAR10_With_Index(root='./datas/data', train=True, download=True, transform=transforms)
            test_dataset = CIFAR10_With_Index('./datas/data', train=False, transform=transform_test)
        else:
            train_dataset = torchvision.datasets.CIFAR10(root='./datas/data', train=True, download=True, transform=transforms)
            test_dataset = torchvision.datasets.CIFAR10('./datas/data', train=False, transform=transform_test)
        img_num_list = [args.num_meta] * args.num_classes
        num_classes = 10

    if args.dataset == 'cifar100':
        
        if args.with_index:
            train_dataset = CIFAR100_With_Index(root='./datas/data', train=True, download=True, transform=transforms)
            test_dataset = CIFAR100_With_Index('./datas/data', train=False, transform=transform_test)
        else:
            train_dataset = torchvision.datasets.CIFAR100(root='./datas/data', train=True, download=True, transform=transforms)
            test_dataset = torchvision.datasets.CIFAR100('./datas/data', train=False, transform=transform_test)
        img_num_list = [args.num_meta] * args.num_classes
        num_classes = 100
    if num_classes > args.num_classes:
        class_list = np.random.randint(0,num_classes,size=args.num_classes)
        train_dataset = get_sub_class_dataset(train_dataset,class_list,num_classes,reset_index=True)
        test_dataset = get_sub_class_dataset(test_dataset,class_list,num_classes,reset_index=True)
    elif num_classes < args.num_classes:
        print("args.num_classes is larger than dataset class num")
        exit(1)

    meta_dataset = sample_dataset(train_dataset, img_num_list, args.num_classes, 'select')
    np.random.seed(args.seed)
    train_dataset=sample_dataset(train_dataset,img_num_list,args.num_classes,'delete')
    return meta_dataset,train_dataset,test_dataset

def get_imbalance_dataset(args,dataset):
    img_num_list = get_img_num_per_cls(args.dataset, args.num_classes, args.imb_factor, args.num_meta)
    print("img_num_list:", img_num_list)
    print("img_num_list-sum:", sum(img_num_list))
    imbalance_dataset=sample_dataset(dataset,img_num_list,args.num_classes,'select')
    return imbalance_dataset


def get_img_num_per_cls(dataset,cls_num,imb_factor=None,num_meta=None):
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
    if dataset == 'cifar10':
        img_max = 5000-num_meta

    if dataset == 'cifar100':
        img_max = 500-num_meta

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    imbalance_ratios = get_imbalance_ratios(imb_factor, cls_num)
    for cls_idx in range(cls_num):
        ratio = imbalance_ratios[cls_idx]
        num = img_max * ratio
        img_num_per_cls.append(int(num))
    return img_num_per_cls


def sample_dataset(dataset, img_num_list, num_classes, kind):
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
        data_list[j] = [i for i, label in enumerate(dataset.targets) if label == j]

    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        '''
        cls_idx : class index
        img_id_list:sample global index list
        data_list:{'cls_idx':[img_id_list],}
        '''
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        if kind=='delete':
            idx_to_del.extend(img_id_list[:img_num])
        else:
            idx_to_del.extend(img_id_list[img_num:])

    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = np.delete(dataset.targets, idx_to_del, axis=0)
    new_dataset.data = np.delete(dataset.data, idx_to_del, axis=0)

    return new_dataset


def sample_dataset_with_index(dataset, data_index):
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = dataset.target[data_index]
    new_dataset.data = dataset.data[data_index]
    return new_dataset

def sample_dataset_with_indexs_trans(dataset, select_index,select_prob,transform_label,transform_unlabel):
    new_dataset_1 = copy.deepcopy(dataset)    
    new_dataset_2 = copy.deepcopy(dataset)
    new_dataset_1.targets = dataset.target[select_index]
    new_dataset_1.data = dataset.data[select_index]
    new_dataset_1.data_index = dataset.data_index[select_index]
    new_dataset_1.transform = transform_label
    new_dataset_1.mode = 'label'
    new_dataset_1.prob = select_prob
    
    new_dataset_2.targets = np.delete(dataset.targets, select_index, axis=0)
    new_dataset_2.data = np.delete(dataset.data, select_index, axis=0)
    new_dataset_2.data_index = np.delete(dataset.data_index, select_index, axis=0)
    new_dataset_2.transform = transform_unlabel
    new_dataset_2.mode = 'unlabel'
    
    return new_dataset_1,new_dataset_2

def get_sub_class_dataset(dataset, class_list, num_classes,reset_index=False):
    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(dataset.targets) if label == j]
    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        if cls_idx not in class_list:
            idx_to_del.extend(img_id_list)
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = np.delete(dataset.targets, idx_to_del, axis=0)
    new_dataset.data = np.delete(dataset.data, idx_to_del, axis=0)
    if reset_index:
        # convert discrete label to continuous label
        label_convertor = LabelEncoder()
        label_convertor.fit(class_list)
        new_dataset.targets = label_convertor.transform(new_dataset.targets)
    return new_dataset

def get_single_class_dataset(dataset,class_index):
    data_list = [i for i, label in enumerate(dataset.targets) if label == class_index]
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = dataset.targets[data_list]
    new_dataset.data = dataset.data[data_list]
    new_dataset.data_index = dataset.data_index[data_list]
    return new_dataset

def get_other_class_dataset(dataset,class_index):
    data_list = [i for i, label in enumerate(dataset.targets) if label == class_index]
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = np.delete(dataset.targets, data_list, axis=0)
    new_dataset.data = np.delete(dataset.data, data_list, axis=0)
    new_dataset.data_index = np.delete(dataset.data_index, data_list, axis=0)
    return new_dataset

def get_sub_dataset(dataset,sub_index):
    new_dataset = copy.deepcopy(dataset)
    new_dataset.targets = dataset.targets[sub_index]
    new_dataset.data = dataset.data[sub_index]
    return new_dataset


def get_only_noisy_dataset(clean_dataset, with_noisy_dataset):
    idx_to_del = []
    # print(clean_dataset.targets)
    for idx in range(0,len(clean_dataset)):
        if clean_dataset.targets[idx] == with_noisy_dataset.targets[idx]:
            idx_to_del.append(idx)
    new_dataset = copy.deepcopy(with_noisy_dataset)
    new_dataset.targets = np.delete(with_noisy_dataset.targets, idx_to_del, axis=0)
    new_dataset.data = np.delete(with_noisy_dataset.data, idx_to_del, axis=0)
    return new_dataset

def get_only_clean_dataset(clean_dataset, with_noisy_dataset):
    idx_to_del = []
    for idx in range(0,len(clean_dataset)):
        if clean_dataset.targets[idx] == with_noisy_dataset.targets[idx]:
            idx_to_del.append(idx)
    new_dataset = copy.deepcopy(with_noisy_dataset)
    new_dataset.targets = with_noisy_dataset.targets[idx_to_del]
    new_dataset.data = with_noisy_dataset.data[idx_to_del]
    return new_dataset

def get_sub_clean_or_noisy_dataset(clean_dataset, with_noisy_dataset,class_index):
    idx_to_del_1 = []
    idx_to_del_2 = []

    for idx in range(0,len(clean_dataset)):
        if clean_dataset.targets[idx]==class_index:
            idx_to_del_1.append(idx)
            if clean_dataset.targets[idx] == with_noisy_dataset.targets[idx] :
                idx_to_del_2.append(idx)
                idx_to_del_1.remove(idx)
    clean_new_dataset = copy.deepcopy(clean_dataset)
    noisy_new_dataset = copy.deepcopy(with_noisy_dataset)
    clean_new_dataset.targets = clean_dataset.targets[idx_to_del_2]
    clean_new_dataset.data = clean_dataset.data[idx_to_del_2]
    noisy_new_dataset.targets = with_noisy_dataset.targets[idx_to_del_1]
    noisy_new_dataset.data = with_noisy_dataset.data[idx_to_del_1]
    return clean_new_dataset,noisy_new_dataset

def get_sub_clean_or_noisy_dataset_2(clean_dataset, with_noisy_dataset,class_index):
    idx_to_del_1 = []
    idx_to_del_2 = []
    for idx in range(0,len(clean_dataset)):
        if with_noisy_dataset.targets[idx]==class_index:
            idx_to_del_1.append(idx)
            if clean_dataset.targets[idx] == with_noisy_dataset.targets[idx] :
                idx_to_del_2.append(idx)
                idx_to_del_1.remove(idx)
    clean_new_dataset = copy.deepcopy(clean_dataset)
    noisy_new_dataset = copy.deepcopy(with_noisy_dataset)
    clean_new_dataset.targets = clean_dataset.targets[idx_to_del_2]
    clean_new_dataset.data = clean_dataset.data[idx_to_del_2]
    noisy_new_dataset.targets = with_noisy_dataset.targets[idx_to_del_1]
    noisy_new_dataset.data = with_noisy_dataset.data[idx_to_del_1]
    return clean_new_dataset,noisy_new_dataset

def get_sub_clean_or_noisy_dataset_3(clean_dataset, with_noisy_dataset,class_index):
    idx_to_del_1 = []
    # print(clean_dataset.targets)
    for idx in range(0,len(clean_dataset)):
        if with_noisy_dataset.targets[idx]==class_index:
            idx_to_del_1.append(idx)
    new_dataset = copy.deepcopy(clean_dataset)
    new_dataset.targets = clean_dataset.targets[idx_to_del_1]
    new_dataset.data = clean_dataset.data[idx_to_del_1]
    return new_dataset

def get_imbalance_ratios(imb_factor, cls_num):
    imbalance_ratios = []
    for cls_idx in range(cls_num):
        ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
        imbalance_ratios.append(ratio)
    return imbalance_ratios




def get_inverse_imbalance_sampler(args, data):
    # init weights list
    weights = torch.zeros(len(data), dtype=torch.long)

    # get imbalance ratio
    sample_probability = get_imbalance_ratios(imb_factor=args.imb_factor, cls_num=args.num_classes)
    # get reverse
    sample_probability.sort(reverse=False)
    torch_sample_probability = torch.tensor(sample_probability)
    lables = []
    # give sample weight
    for index, (data, target) in enumerate(data):
        lables.append(target)
    weights = torch_sample_probability[lables]
    # create  inverse_imbalance_sampler
    inverse_imbalance_sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(data), replacement=True)
    return inverse_imbalance_sampler



def uniform_mix_c(mixing_ratio, num_classes):
    """
    returns a linear interpolation of a uniform matrix and an identity matrix
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_c(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    torch.save(C, 'noisy_transaction_matrix_real.pt')
    return C


def flip_labels_c_two(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C

def circle_flip_label(corruption_prob, num_classes):
    C_1 = np.eye(num_classes)
    C_2 = np.fliplr(C_1)
    return C_1 * (1 - corruption_prob) + C_2 * corruption_prob

def manual_label(corruption_prob, num_classes,seed=1):
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    C[0][1] = corruption_prob
    C[1][0] = corruption_prob
    for i in range(2,num_classes):
        C[i][np.random.choice(row_indices[(row_indices != i ) & (row_indices > 1)])] = corruption_prob
    torch.save(C, 'noisy_transaction_matrix_real.pt')
    return C

def get_noisy_dataset(dataset, args):
    # avoid make influence on origin dataset
    new_dataset = copy.deepcopy(dataset)
    if args.corruption_type == 'unif':
        noisy_transaction_matrix_real = uniform_mix_c(args.corruption_prob, args.num_classes)
        print(noisy_transaction_matrix_real)
    elif args.corruption_type == 'flip':
        # noisy_transaction_matrix_real = flip_labels_c(args.corruption_prob, args.num_classes,seed=args.seed)
        noisy_transaction_matrix_real = flip_labels_c(args.corruption_prob, args.num_classes)
        print(noisy_transaction_matrix_real)
    elif args.corruption_type == 'flip2':
        noisy_transaction_matrix_real = flip_labels_c_two(args.corruption_prob, args.num_classes,seed=args.seed)
        print(noisy_transaction_matrix_real)
    elif args.corruption_type == 'cflip':
        noisy_transaction_matrix_real = circle_flip_label(args.corruption_prob, args.num_classes)
        print(noisy_transaction_matrix_real)
    elif args.corruption_type == 'manual':
        # noisy_transaction_matrix_real = flip_labels_c(args.corruption_prob, args.num_classes,seed=args.seed)
        noisy_transaction_matrix_real = manual_label(args.corruption_prob, args.num_classes)
        print(noisy_transaction_matrix_real)
    else:
        noisy_transaction_matrix_real = None
    for i in range(len(new_dataset.targets)):
        new_dataset.targets[i] = np.random.choice(args.num_classes, p=noisy_transaction_matrix_real[new_dataset.targets[i]])
    return new_dataset, noisy_transaction_matrix_real

