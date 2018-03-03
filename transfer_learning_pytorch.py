# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 23:01:20 2017
reference: https://zhuanlan.zhihu.com/p/30119664
           https://gist.github.com/panovr/2977d9f26866b05583b0c40d88a315bf

@author: Sampson
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from torchvision import datasets, models, transforms
from sklearn.metrics.pairwise import pairwise_distances
import os, datetime, time, random
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/lixinpeng/pretrained-models.pytorch')
import pretrainedmodels
import pretrainedmodels.utils

def FineTuneModel(model, num_classes, tune_ratio, args=None):
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, num_classes)
    
    if args.arch=='bninception':
        fea_layer_num = len(list(model.children())[:-1])
        for layer in list(model.children())[:-int(fea_layer_num*tune_ratio+1)]:
            for p in layer.parameters():
                p.requires_grad = False
    else:
        fea_layer_num = len(list(model._features.children()))
        for layer in list(model._features.children())[:-int(fea_layer_num*tune_ratio)]:
            for p in layer.parameters():
                p.requires_grad = False
    return model
    
def train(model, train_loader, args):
    model.train()
    criterion = nn.CrossEntropyLoss()
    all_paras = filter(lambda p: (p.requires_grad), model.parameters())
    cls_paras = list(model.last_linear.parameters())
    fea_paras = set(all_paras)-set(cls_paras)
    optimizer = optim.SGD([
            {'params': fea_paras, 'lr': args.fea_lr},
            {'params': cls_paras, 'lr': args.cls_lr}
            ], momentum=0.9, weight_decay=0.0005)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    since = time.time();log=[]
    for epoch in range(args.epochs):
        if epoch==0: torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch0.pth'))
        exp_lr_scheduler.step()
        running_loss = 0.0; running_corrects = 0.0
        
        for i, [inputs, labels] in enumerate(train_loader):
            # if i==3: break;
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            _, preds = torch.max(outputs, 1)
            running_corrects += ((preds==labels).type(torch.FloatTensor).sum() / labels.size(0)).data[0]
              
        epoch_loss = running_loss / (i+1)
        epoch_acc = running_corrects / (i+1)
        log.append([epoch, epoch_loss, epoch_acc])
        
        print('Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, epoch_loss, epoch_acc))
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch{}.pth'.format(epoch+1)))            
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    log = np.array(log)
    plt.figure()
    plt.plot(log[:,0], log[:,1], label='loss')
    plt.plot(log[:,0], log[:,2], label='acc')
    plt.legend(loc='upper right');plt.xlabel("Epoch");plt.ylabel("Loss");
    img_dir = os.path.join(args.save_dir, 
            'train_'+args.arch+'_'+str(args.tune_ratio)+'_'+str(args.fea_lr)+'_'+str(args.cls_lr)+'.png')
    plt.savefig(img_dir, bbox_inches='tight')
#    plt.show()
    return 'train successfully'

def gallery_probe_split(data, sample_num):
    labels = data['label']
    feas = data['feature']
    idset = set(labels)
    gallery_feas=[];gallery_labs=[];probe_feas=[];probe_labs=[];
    for id_ in idset:
        mask = (labels==id_)
        labels_ = labels[mask]
        feas_ = feas[mask]
        indices = random.sample(range(feas_.shape[0]), sample_num)
        gallery_feas.append(feas_[indices]);probe_feas.append(np.delete(feas_,indices, axis=0));
        gallery_labs.append(labels_[indices]);probe_labs.append(np.delete(labels_, indices, axis=0));
        
    gallery_feas = np.concatenate(gallery_feas, axis=0)
    gallery_labs = np.concatenate(gallery_labs, axis=0)
    probe_feas = np.concatenate(probe_feas, axis=0)
    probe_labs = np.concatenate(probe_labs, axis=0)
    gallery={'feature':gallery_feas, 'label':gallery_labs}
    probe={'feature':probe_feas, 'label':probe_labs}
    return gallery, probe
        
def iden(model, val_loader, args):
    log = []; best_acc = [0.0, 0.0, 0.0];
    for epoch in range(args.epochs+1):
        try:
            model.load_state_dict(torch.load(os.path.join(args.save_dir, 'epoch{}.pth'.format(epoch))))
            print('succeed to load model ' + args.arch)
        except:
            print('fail to load model ' + args.arch)
            return 'fail to load model'
        model.eval()
        
        all_features=[];all_labels=[]
        for i, [batch_image, batch_id_label] in enumerate(val_loader):
            batch_image = Variable(batch_image.cuda())
            batch_features = model.features(batch_image).cpu().data.numpy()
            batch_features = batch_features.reshape((batch_features.shape[0], -1))
            all_features.append(batch_features)
            all_labels.append(np.array(batch_id_label))
            
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        fea_lab={'label':all_labels, 'feature':all_features}
        
        for gallery_num in range(3):
            epoch_acc=0
            for iter in range(20):
                gallery, probe=gallery_probe_split(fea_lab, gallery_num+1)
                print(fea_lab['feature'].shape);print(gallery['feature'].shape);print(probe['feature'].shape);
                Distance = pairwise_distances(gallery['feature'], probe['feature'], metric='cosine', n_jobs=-1)
                maxindex = np.argsort(Distance, axis=0)[0,:]
    #            print(maxindex)
                match = (gallery['label'][maxindex] == probe['label'])
                print(match)
                epoch_acc += float(match.sum())/match.shape[0]
            epoch_acc = epoch_acc/20
            log.append([epoch, epoch_acc])
            print('Acc: {:.4f}'.format(epoch_acc))
            if epoch_acc > best_acc[gallery_num]: best_acc[gallery_num] = epoch_acc
                                   
    log = np.array(log).reshape(-1, 2*3)
    plt.figure()
    for gallery_num in range(3):
        plt.plot(log[:,gallery_num*2], log[:,gallery_num*2+1], label=str(gallery_num)) 
    plt.legend(loc='upper right');plt.title('{}|{}|{}'.format(best_acc[0],best_acc[1],best_acc[2]));
    img_dir = os.path.join(args.save_dir, 
            'iden_'+args.arch+'_'+str(args.tune_ratio)+'_'+str(args.fea_lr)+'_'+str(args.cls_lr)+'.png')
    plt.savefig(img_dir, bbox_inches='tight')
#    plt.show()
    return 'identify successfully'

def load_data(data_dir, model):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            pretrainedmodels.utils.TransformImage(model)
        ]),
        'val': pretrainedmodels.utils.TransformImage(model),
    }
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('we have {} train data and {} val data'.format(dataset_sizes['train'], dataset_sizes['val']))
    samples, _ = next(iter(dataloders['train']))
    vutils.save_image(samples, './samples_train.png', normalize=True)

    return image_datasets, dataloders

def main(args):
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)

    original_model = pretrainedmodels.models.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    
    image_datasets, dataloders = load_data(args.data_dir, original_model)
    num_classes = len(image_datasets['train'].classes)
    print("using pre-trained model '{}' and transform to {} classes".format(args.arch, num_classes))
    model = FineTuneModel(original_model, num_classes, args.tune_ratio, args)
    print(args.arch)
#    torch.save(original_model.state_dict(), os.path.join(args.save_dir, 'epoch0.pth'))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if len(args.device.split(','))>1:
        print('model uses multigpu!')
        model = nn.DataParallel(model)
    model.cuda()
    
    message = train(model, dataloders['train'], args)
    print(message)
    message = iden(model, dataloders['val'], args)
    print(message)

class args(object):
    epochs=20
    arch='bninception'
    fea_lr=0.0001
    cls_lr=0.001
    tune_ratio=0.5
    device='1'
    save_dir='./'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#    save_dir='/data/lixinpeng/analyzing-chimpanzees/2018-01-02_23-00-53/'
    data_dir='/data/lixinpeng/DataBase/Animal_MarryAnne_230jpg/'

# enjoy alchemy
datestr = './'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
archlist=['alexnet', 'resnet', 'vgg16', 'densenet121', 'bninception']
log_list = range(1,7) # coarse_scale
ratio_list = np.linspace(0.25, 1, 4)
fea_lr_list= np.linspace(1e-3, 1e-5, 5)
cls_lr_list= np.linspace(1e-3, 1e-5, 5)
main(args)
#for arch in archlist:
#for ratio in ratio_list:
#    for fea_lr in fea_lr_list:
#        for cls_lr in cls_lr_list:
#            args.tune_ratio = ratio
#            args.fea_lr = fea_lr
#            args.cls_lr = cls_lr
#            main(args)