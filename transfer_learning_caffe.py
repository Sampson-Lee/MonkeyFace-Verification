# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 23:01:20 2017

@author: Sampson
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import random
import matplotlib.pyplot as plt
import caffe

# 对输入数据做相应地调整如通道、尺寸等等
def preprocess(image):
    image = image[19:211, 19:211, :]
    transformer = caffe.io.Transformer({'data': img_size})
    transformer.set_transpose('data', (2,0,1))
    mu = np.array([73, 66, 42])
    transformer.set_mean('data', mu)   # substracte pixel
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    return transformer.preprocess('data', image)
    
# 提取特征并保存为相应地文件
def extractFeature(imageList, net):
    # set net to batch size of 1 如果图片较多就设置合适的batchsize
    net.blobs['data'].reshape(1, 3, 192, 192)
    with open(imageList, 'r') as file:
        num = 0; fea_list = []; label_list = [];
        for line in file.readlines():
            imgPath, label = line.strip().rstrip('\n').split(' ')
            net.blobs['data'].data[...] = preprocess(caffe.io.load_image(imgPath))
            out = net.forward()
            fea = net.blobs[fea_blob].data[0].flatten()
            fea_list.append(fea); label_list.append(label)
            num += 1
            if num%20==0:
                print(net.blobs[fea_blob].data[0].shape)
                print('Num ',num,' extract feature ', fea)
        fea_list = np.array(fea_list);label_list = np.array(label_list);
    print(num)
    return fea_list, label_list

def gallery_probe_split(feas, labels, sample_num):
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

def test(model_list, model_test, save_log):
    log = []; best_acc = [0.0, 0.0, 0.0];
    with open(model_list, 'r') as file:
        models = file.readlines()
    for i in range(len(models)):
        net = caffe.Net(model_test, models[i].strip().rstrip('\n'), caffe.TEST)
        feas, labels = extractFeature(imageList, net)
        for gallery_num in range(3):
            epoch_acc=0
            for iter in range(20):
                gallery, probe=gallery_probe_split(feas, labels, gallery_num+1)
                print(feas.shape);print(gallery['feature'].shape);print(probe['feature'].shape);
                Distance = pairwise_distances(gallery['feature'], probe['feature'], metric='cosine', n_jobs=-1)
                maxindex = np.argsort(Distance, axis=0)[0,:]
                match = (gallery['label'][maxindex] == probe['label'])
                print(match)
                epoch_acc += float(match.sum())/match.shape[0]
            epoch_acc = epoch_acc/20
            log.append([i, epoch_acc])
            print('Acc: {:.4f}'.format(epoch_acc))
            if epoch_acc > best_acc[gallery_num]: best_acc[gallery_num] = epoch_acc
                       
    log = np.array(log).reshape(-1, 2*3)
    plt.figure()
    for gallery_num in range(3):
        plt.plot(log[:,gallery_num*2], log[:,gallery_num*2+1], label=str(gallery_num)) 
    plt.legend(loc='upper right');plt.title('{}|{}|{}'.format(best_acc[0],best_acc[1],best_acc[2]));
    plt.savefig(save_log, bbox_inches='tight')

def train(model_solver, save_log, weights_dir):
    solver = caffe.get_solver(model_solver)
    solver.net.copy_from(weights_dir)
#    solver.solve() # train the network according solver
    niter = 6625; display = 265; _train_loss = 0; train_loss = np.zeros(int(niter * 1.0 / display));
    for it in range(niter):
        # 进行一次解算
        solver.step(1)
        # 统计train loss  
        _train_loss += solver.net.blobs['loss'].data  
        if it % display == 0:
            # 计算平均train loss  
            train_loss[it // display] = _train_loss / display
            print(train_loss[it // display])
            _train_loss = 0                 
    print('\nplot the train loss and test accuracy\n')
    plt.figure()
    plt.plot(display * range(len(train_loss)), train_loss, label='trainloss') 
    plt.savefig(save_log, bbox_inches='tight')
    
finetune_type = 'inception-v2-face_0.75_0.0001_0.001'
model_dir = '/data/lixinpeng/analyzing-chimpanzees/inception-v2-face/'
gpuID = 3
fea_blob = 'pool5/2x2_s1'

imageList = '/data/lixinpeng/DataBase/Animal_MarryAnne_230jpg/val_list.txt'
img_size = (1, 3, 192, 192)

if __name__ == "__main__":
    caffe.set_mode_gpu()
    caffe.set_device(gpuID)
    train(model_solver=model_dir+'solver_04.prototxt', 
          save_log=model_dir+finetune_type+'_train.png', 
          weights_dir=model_dir+'PaSC_Challenge_Model04_GoogLeNetTune_iter_21000.caffemodel')
    
    test(model_list=model_dir+'model_list.txt', 
         model_test=model_dir+'deploy_04.prototxt', 
         save_log=model_dir+finetune_type+'_iden.png')