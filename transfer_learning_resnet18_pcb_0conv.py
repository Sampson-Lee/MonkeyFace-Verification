# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:35:01 2017
reference: https://zhuanlan.zhihu.com/p/30119664
           https://gist.github.com/panovr/2977d9f26866b05583b0c40d88a315bf

@author: Sampson
"""

""" face aligned points
transform function: (x-15, y-15)x(230/200)x(14/230)

[   94  116; -> 5 7  -> (2:9, 4:11)     eye_left
   137  116; -> 8 7  -> (5:12, 4:11)    eye_right
 115.5  142; -> 7 9  -> (4:11, 6:13)    nose
    96  164; -> 6 10 -> (3:10, 7:14)    mouth_left
   135  164] -> 8 10 -> (5:12, 7:14)    mouth_right

but i just choose transforms.FiveCrop as augmentation causes misalignment
"""    

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import torchvision.utils as vutils
from torchvision import datasets, models, transforms
from sklearn.metrics.pairwise import pairwise_distances
import os, datetime, time, random
import matplotlib.pyplot as plt

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(200),
        transforms.Resize(230),
#        transforms.CenterCrop(224),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(),
#        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(200),
        transforms.Resize(230),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class FineTuneResNet(nn.Module):
    def __init__(self, arch, num_classes, tune_ratio):
        super(FineTuneResNet, self).__init__()

        import torchvision.models.resnet as model
        import torch.utils.model_zoo as model_zoo

        resnet = models.__dict__[arch](pretrained=False)
        if arch=='resnet18':
            resnet.inplanes = 256
            resnet.layer4 = resnet._make_layer(model.BasicBlock, planes=512, blocks=2, stride=1)
            resnet.load_state_dict(model_zoo.load_url(model.model_urls['resnet18']))
        if arch=='resnet50':
            resnet.inplanes = 1024
            resnet.layer4 = resnet._make_layer(model.Bottleneck, planes=512, blocks=2, stride=1)
            resnet.load_state_dict(model_zoo.load_url(model.model_urls['resnet50']))
            
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc_eye_left = nn.Linear(512, num_classes)
        self.fc_eye_right = nn.Linear(512, num_classes)
        self.fc_nose = nn.Linear(512, num_classes)
        self.fc_mouth_left = nn.Linear(512, num_classes)
        self.fc_mouth_right = nn.Linear(512, num_classes)

        init.kaiming_normal(self.fc_eye_left.weight.data, a=0, mode='fan_in')
        init.kaiming_normal(self.fc_eye_right.weight.data, a=0, mode='fan_in')
        init.kaiming_normal(self.fc_nose.weight.data, a=0, mode='fan_in')
        init.kaiming_normal(self.fc_mouth_left.weight.data, a=0, mode='fan_in')
        init.kaiming_normal(self.fc_mouth_right.weight.data, a=0, mode='fan_in')
        
        # Freeze some layer weights
        fea_layer_num = len(list(self.features.children()))
        for layer in list(self.features.children())[:-int(fea_layer_num*tune_ratio)]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, x, extractG=False, extractH=False):
        batch_size = x.size(0)
        f = self.features(x)
#        print(f.size())
        tensorT = torch.cat((f[:,:,2:9,4:11], f[:,:,5:12,4:11], \
                             f[:,:,4:11,6:13], f[:,:,3:10,7:14], f[:,:,5:12,7:14]), dim=3) # (n, 512, 7, 7*5)
#        print(tensorT.size())
        
        tensorG = self.avg_pool(tensorT) #(n, 512, 1, 5)
        if extractG: return tensorG
#        print(tensorG.size())
        
#        tensorH = self.conv1x1(tensorG)
        tensorH = tensorG
        if extractH: return tensorH
#        print(tensorH.size())
        
        eye_left = tensorH[:,:,:,0].contiguous().view(batch_size, -1)
        eye_right = tensorH[:,:,:,1].contiguous().view(batch_size, -1)
        nose = tensorH[:,:,:,2].contiguous().view(batch_size, -1)
        mouth_left = tensorH[:,:,:,3].contiguous().view(batch_size, -1)
        mouth_right = tensorH[:,:,:,4].contiguous().view(batch_size, -1)
        
        pre_eye_left = self.fc_eye_left(eye_left)
        pre_eye_right = self.fc_eye_right(eye_right)
        pre_nose = self.fc_nose(nose)
        pre_mouth_left = self.fc_mouth_left(mouth_left)
        pre_mouth_right = self.fc_mouth_right(mouth_right)
#        print(pre_eye_left.size())
        
        return pre_eye_left, pre_eye_right, pre_nose, pre_mouth_left, pre_mouth_right

def train(model, train_loader, args):
    model.train()
    criterion = nn.CrossEntropyLoss(size_average=False)
#    criterion = nn.CrossEntropyLoss(size_average=True)
    all_paras = filter(lambda p: (p.requires_grad), model.parameters())
    fea_paras = filter(lambda p: (p.requires_grad), model.features.parameters())
    cls_paras = set(all_paras)-set(fea_paras)
    optimizer = optim.SGD([
            {'params': fea_paras, 'lr': args.fea_lr},
            {'params': cls_paras, 'lr': args.cls_lr}
            ], momentum=0.9, weight_decay=0.0005)
    milestones=[int(args.epochs*(0.5**multi)) for multi in reversed(range(3))]
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    since = time.time();log=[];
    for epoch in range(args.epochs):
        if epoch==0: torch.save(model.state_dict(), os.path.join(args.save_dir, 'epoch0.pth'))
        exp_lr_scheduler.step()
        running_loss = 0.0; running_corrects = 0.0; img = 0;
        
        for i, [inputs, labels] in enumerate(train_loader):
            # if i==3: break;
            img += labels.size(0)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            
            optimizer.zero_grad()

            pre_eye_left, pre_eye_right, pre_nose, pre_mouth_left, pre_mouth_right = model(inputs)
            loss = criterion(pre_eye_left, labels)+criterion(pre_eye_right, labels) \
                    + criterion(pre_nose, labels) + criterion(pre_mouth_left, labels) + criterion(pre_mouth_right, labels)
            loss.backward()
            optimizer.step()
            
#            print(labels.size())
            running_loss += loss.data[0]
            _, preds = torch.max(pre_eye_left, 1)
            running_corrects += (preds==labels).type(torch.FloatTensor).sum().data[0]
            _, preds = torch.max(pre_eye_right, 1)
            running_corrects += (preds==labels).type(torch.FloatTensor).sum().data[0]
            _, preds = torch.max(pre_nose, 1)
            running_corrects += (preds==labels).type(torch.FloatTensor).sum().data[0]
            _, preds = torch.max(pre_mouth_left, 1)
            running_corrects += (preds==labels).type(torch.FloatTensor).sum().data[0]
            _, preds = torch.max(pre_mouth_right, 1)
            running_corrects += (preds==labels).type(torch.FloatTensor).sum().data[0]
            
        epoch_loss = running_loss / (img*5)
        epoch_acc = running_corrects / (img*5)
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
                           'train_'+'_'+args.lr_policy+args.preprocess+'_bs'+str(args.batch_size)+'_'+args.arch+'_ratio'+ \
                            str(args.tune_ratio)+'_fea_lr'+str(args.fea_lr)+'_cls_lr'+str(args.cls_lr)+'.png')
    plt.savefig(img_dir, bbox_inches='tight')
#    plt.show()
    return 'train successfully'

def gallery_probe_split(feas, labels, sample_num):
#    labels = data['label']
#    feas = data['feature']
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
        
        all_features=[];all_labels=[];
        for i, [batch_image, batch_id_label] in enumerate(val_loader):
            batch_image = Variable(batch_image.cuda())
            batch_features = model(batch_image, extractG=args.extractG, extractH=args.extractH).cpu().data.numpy() # tensorG=(bs,512,1,5)
#            batch_features = model(batch_image, extractG=True).cpu().data.numpy() # tensorH=(n,256,1,5)
            all_features.append(batch_features)
            all_labels.append(np.array(batch_id_label))
            
        all_features = np.concatenate(all_features, axis=0) # (n,512,1,5)
        all_labels = np.concatenate(all_labels, axis=0) # (n,)
#        fea_lab={'label':all_labels, 'feature':all_features}
        
        allnum, _, _, partnum = all_features.shape
        
        for gallery_num in range(3):
            epoch_acc=0.0;
            for iter in range(20):
                # method one: every part votes for label
                if args.method=='part':
                    print('use method part!')
                    for part in range(partnum):
                        features = all_features[:,:,:,part].reshape(allnum, -1)
                        gallery, probe=gallery_probe_split(features, all_labels, gallery_num+1)
                        print(features.shape);print(gallery['feature'].shape);print(probe['feature'].shape);
                        if part==0:
                            Distance = pairwise_distances(gallery['feature'], probe['feature'], metric='cosine', n_jobs=-1)
                        else:
                            Distance += pairwise_distances(gallery['feature'], probe['feature'], metric='cosine', n_jobs=-1)
                
                # method two: concatenate all parts
                if args.method=='concat':
                    print('use method concat!')
                    features = all_features.reshape(allnum, -1)
                    gallery, probe=gallery_probe_split(features, all_labels, gallery_num+1)
                    print(features.shape);print(gallery['feature'].shape);print(probe['feature'].shape);
                    Distance = pairwise_distances(gallery['feature'], probe['feature'], metric='cosine', n_jobs=-1)                
                
                maxindex = np.argsort(Distance, axis=0)[0,:]
                match = (gallery['label'][maxindex] == probe['label'])
                print(match)
                epoch_acc += float(match.sum())/match.shape[0]
                
            epoch_acc = round(epoch_acc/20, 4)
            log.append([epoch, epoch_acc])
            print('epoch_acc: {:.4f} '.format(epoch_acc))
            if epoch_acc > best_acc[gallery_num]: best_acc[gallery_num] = epoch_acc
    
    log = np.array(log).reshape(-1, 2*3)
    plt.figure()
    for gallery_num in range(3):
        plt.plot(log[:,gallery_num*2], log[:,gallery_num*2+1], label=str(gallery_num+1)) 
    plt.legend(loc='upper right');plt.title('{}|{}|{}'.format(best_acc[0],best_acc[1],best_acc[2]));
    img_dir = os.path.join(args.save_dir, 
                           'iden_'+args.lr_policy+'_'+args.preprocess+'_bs'+str(args.batch_size)+'_'+args.arch+'_ratio'+ \
                            str(args.tune_ratio)+'_fea_lr'+str(args.fea_lr)+'_cls_lr'+str(args.cls_lr)+ \
                            '_extractG_'+str(args.extractG)+'_extractH_'+str(args.extractH)+'_'+args.method+'.png')
    plt.savefig(img_dir, bbox_inches='tight')
#    plt.show()
    return 'identify successfully'

# training set：230->centercrop->200->resize->230->augmentation->224
# testing set：230->centercrop->200->resize->230->centercrop->224
def load_data(data_dir, batch_size):
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('we have {} train data and {} val data, iteration {} per epoch'.format(dataset_sizes['train'], dataset_sizes['val'], int(dataset_sizes['train']/batch_size)))
    samples, _ = next(iter(dataloders['train']))
    vutils.save_image(samples, './samples_train.png', normalize=True)

    return image_datasets, dataloders

def main(args):
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)

    image_datasets, dataloders = load_data(args.data_dir, args.batch_size)
    num_classes = len(image_datasets['train'].classes)
    
    print("using pre-trained model '{}' and transform to {} classes".format(args.arch, num_classes))
    model = FineTuneResNet(args.arch, num_classes, args.tune_ratio)
    print(args.arch)
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if len(args.device.split(','))>1:
        print('model uses multigpu!')
        model = nn.DataParallel(model)
    model.cuda()
    
    message = train(model, dataloders['train'], args)
    print(message)
    for method in method_list:
        for extractG in extractG_list:
            args.method = method
            args.extractG = extractG
            args.extractH = not extractG
            message = iden(model, dataloders['val'], args)
    print(message)

class args(object):
    preprocess='crop200_sca230_jitter_crop_pcb_0conv'
    epochs=30
    method='part'
    extractG = True
    extractH = False
    arch='resnet18'
    batch_size = 128
    fea_lr=1e-4
    cls_lr=5.005e-5
    lr_policy='multistep'
    tune_ratio=0.75
    device='0'
    save_dir='./'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#    save_dir='/home/xinpeng/analyzing-chimpanzees/2018-01-10_12-39-38/'
    data_dir='/data/dataset/Monkey/Animal_MarryAnne_230jpg/'

# enjoy alchemy
datestr = './'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
archlist=['alexnet', 'resnet', 'vgg16', 'densenet121', 'bninception']
log_list = range(1,7) # coarse_scale
ratio_list = [0.6, 0.75, 0.85]# np.linspace(0.25, 1, 4)
bs_list = [128, 150, 80]
method_list = ['part', 'concat']
extractG_list = [True, False]

fea_lr_list= np.linspace(1e-5, 1e-7, 5)
cls_lr_list= np.linspace(1e-4, 1e-6, 5)
#main(args)
for bs in bs_list:
    for ratio in ratio_list:
        for fea_lr in fea_lr_list:
            for cls_lr in cls_lr_list:
                args.batch_size = bs
                args.tune_ratio = ratio
                args.fea_lr = fea_lr
                args.cls_lr = cls_lr
                main(args)
    
