# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 21:52:34 2017

@author: Sampson
"""

import os, random
import numpy as np

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics.pairwise import pairwise_distances

from PIL import Image

img_transform = transforms.Compose([
                  transforms.ToTensor(),
			             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class alexnet:
    def __init__(self):
        self.alexnet=models.alexnet(pretrained=True)
        self.alexnet.cuda()#将模型从CPU发送到GPU,如果没有GPU则删除该行

    #特征提取
    def extract_feature(self,imgpath):
        model = self.alexnet
        model.classifier=torch.nn.LeakyReLU(1)
        model.eval()

        img=Image.open(imgpath)
        img=img.resize((224,224))
        tensor=img_transform(img)
        
        tensor=tensor.resize_(1,3,224,224)
        tensor=tensor.cuda()

        result=model(Variable(tensor))
        result_npy=result.data.cpu().numpy()
        
        return result_npy[0]

monkey_name_dic = {
	'Gracia':0, 'Gretel':1, 'Juno':2, 'Trevor':3, 
	'voc_Gretel':4, 'Ima':5, 'Indy':6, 'Icarus':7, 
	'Feb':8, 'Teresia':9, 'Joy':10, 'Indah':11
}

def make_data_svm(img_dir, fea_dir):
    model = alexnet()
    monkeylist = os.listdir(img_dir)
    feas=[];labels=[];
    for monkey in monkeylist:
        img_filelist = os.listdir(img_dir+monkey)
        for imgpath in img_filelist:
            fea = model.extract_feature(img_dir+monkey+'/'+imgpath)
            feas.append(fea)
            labels.append(monkey_name_dic[monkey])
        print(len(feas))

    feas = np.array(feas);labels = np.array(labels);
    print(feas.shape)
    print(labels.shape)
    feas_train, feas_test, labels_train, labels_test = train_test_split(feas, labels, test_size=0.20, random_state=42)
    print(feas_train.shape)
    print(labels_train.shape)
    scio.savemat(fea_dir+'feas_train.mat', {'feature':feas_train,'label':labels_train})
    scio.savemat(fea_dir+'feas_test.mat', {'feature':feas_test,'label':labels_test})
    print('make svm data successfully!')

def load_data_svm(clsfea_dir):
    data_train = scio.loadmat(clsfea_dir+'feas_train.mat')
    data_test = scio.loadmat(clsfea_dir+'feas_test.mat')
    feas_train, feas_test, labels_train, labels_test = \
            data_train['feature'],  data_test['feature'], data_train['label'], data_test['label']
    return feas_train, feas_test, labels_train, labels_test

def cls_svm(clsfea_dir):
    feas_train, feas_test, labels_train, labels_test = load_data_svm(clsfea_dir)
    labels_train = labels_train.reshape(-1)
    labels_test = labels_test.reshape(-1)
    clf = svm.SVC()# one vs one && one vs all?
    clf.fit(feas_train, labels_train)
    print(clf.score(feas_test, labels_test))

def make_data_iden(img_dir, fea_dir):
    model = alexnet()
    monkeylist = os.listdir(img_dir)
    galleryfeas=[];gallerylabs=[];
    probefeas=[];probelabs=[];
    for monkey in monkeylist:
        img_filelist = os.listdir(img_dir+monkey)
        sample_ind = random.randint(0,len(img_filelist))
        for i,img in enumerate(img_filelist):
            fea = model.extract_feature(img_dir+monkey+'/'+img)
            if i==sample_ind:
                galleryfeas.append(fea)
                gallerylabs.append(monkey_name_dic[monkey])
            else:
                probefeas.append(fea)
                probelabs.append(monkey_name_dic[monkey])
        print(len(probefeas))
    
    galleryfeas = np.array(galleryfeas);gallerylabs = np.array(gallerylabs);
    probefeas = np.array(probefeas);probelabs = np.array(probelabs);
    scio.savemat(fea_dir+'gallery.mat', {'feature':galleryfeas,'label':gallerylabs})
    scio.savemat(fea_dir+'probe.mat', {'feature':probefeas,'label':probelabs})
    print('make iden data successfully!')
    
def load_data_iden(idenfea_dir):
    data_gallery = scio.loadmat(idenfea_dir+'feas_train.mat')
    data_probe = scio.loadmat(idenfea_dir+'feas_test.mat')
    feas_gallery, feas_probe, labels_gallery, labels_probe = \
            data_gallery['feature'],  data_probe['feature'], data_gallery['label'], data_probe['label']
    return feas_gallery, feas_probe, labels_gallery, labels_probe

def iden_cos(idenfea_dir):
    feas_gallery, feas_probe, labels_gallery, labels_probe = load_data_iden(idenfea_dir)
    labels_gallery = labels_gallery.reshape(-1)
    labels_probe = labels_probe.reshape(-1)
    distance = pairwise_distances(feas_gallery, feas_probe, metric='cosine', n_jobs=-1)
    maxindex = np.argsort(distance, axis=0)[0,:]
    match = (labels_gallery[maxindex] == labels_probe)
    # print(match)
    print(float(match.sum())/match.shape[0])
   
clsfea_dir = '/data5/lixinpeng/dataset/Monkey/Animal_MarryAnne_230fea/'
clsimg_dir='/data5/lixinpeng/dataset/Monkey/Animal_MarryAnne_230jpg/'

idenimg_dir = '/data5/lixinpeng/dataset/Monkey/Animal_MarryAnne_230jpg/test_jpg/'
idenfea_dir = '/data5/lixinpeng/dataset/Monkey/Animal_MarryAnne_230fea/'
#make_data_svm(clsimg_dir, clsfea_dir)
#cls_svm(clsfea_dir)
make_data_iden(idenimg_dir, idenfea_dir)
iden_cos(idenfea_dir)