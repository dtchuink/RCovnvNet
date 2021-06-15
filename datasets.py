from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
from affine_transforms import Affine
#import progressbar


class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 3500, classification = False, class_choice = None, train = True, transform=None ):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category40.txt')
#         self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.transform = transform
        self.classification = classification

        #read the file and associate each object with a folder name
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
#             dir_point = os.path.join(self.root, self.cat[item], 'points')
#             dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_point = os.path.join(self.root, self.cat[item])
            dir_seg = os.path.join(self.root, self.cat[item])
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.8)]
            else:
                fns = fns[int(len(fns) * 0.8):]

            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.pts')))
                
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))


        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print('--- number of classes')    
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(int(len(self.datapath)/50)):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
#         print("point_set max ", np.max(point_set))
#         print("point_set before ", point_set)
        
        #Standardization
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        dist = np.expand_dims(np.expand_dims(dist, 0), 1)
#         print("dist ", dist)
        point_set = point_set/dist
        
        #Normalization
        

        if(len(point_set)<self.npoints):
            for i in range(self.npoints-len(point_set)):
                point_set = np.vstack((point_set, [0, 0, 0]))
        else:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[choice, :]       
            seg = seg[choice]
        
        point_set = point_set + 1e-5 * np.random.rand(*point_set.shape)   
            

#         choice = np.random.choice(len(seg), self.npoints, replace=True)
#         point_set = point_set[choice, :] 
#         point_set = point_set + 1e-5 * np.random.rand(*point_set.shape)         
#         seg = seg[choice]
        
        point_set = torch.from_numpy(point_set.astype(np.float32))
        seg = torch.from_numpy(seg.astype(np.int64))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        
#         print("point =",point_set)
#         print("shape=", point_set.size())
#         
#         if isinstance(self.transform, (tuple,list)):
#             self.input_transform = transforms.Compose(self.input_transform)
        
        if self.transform is not None:
            point_set = self.transform(point_set)
        
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)
    

if __name__ == '__main__':
    print('test')
    #d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    #print(len(d))
    #ps, seg = d[0]
    #print(ps.size(), ps.type(), seg.size(),seg.type())
    
    
    d = PartDataset(root = '/home/danielle/Documents/3DNeuralNetwork/ModelNet40Converted', classification = True)
#     d = PartDataset(root = '/home/danielle/pyDevelopment/lstm-rnn/src/ln_lstm/shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())
#     point_set, cls = d.__getitem__(1)
#     print('----point_set 1')    
#     print(point_set) 
#     print(cls)
#     point_set, cls = d.__getitem__(2)
#     print('----point_set 2')    
#     print(point_set) 
#     print(cls)
#     point_set, class_label = d[0]
#     print(len(point_set))
#     print('point_set 0')
#     print(point_set[0])
#     print('class_label')
#     print(len(class_label))
#     ps, cls = d[0]
#     print(ps.size(), ps.type(), cls.size(),cls.type())

