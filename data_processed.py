import torch
from torchvision import transforms
import copy
import random
import os
import numpy as np
from PIL import Image
from imageio import imread
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

def filt_small_instance(coco_item, pixthreshold=4000,imgNthreshold=5):
    list_dict = coco_item.catToImgs
    for catid in list_dict:
        list_dict[catid] = list(set( list_dict[catid] ))
    new_dict = copy.deepcopy(list_dict)
    for catid in list_dict:
        imgids = list_dict[catid]
        for n in range(len(imgids)):
            imgid = imgids[n]
            anns = coco_item.imgToAnns[imgid]
            has_large_instance = False
            for ann in anns:
                if (ann['category_id'] == catid) and (ann['iscrowd'] == 0) and (ann['area'] > pixthreshold):
                    has_large_instance = True
            if has_large_instance is False:
                new_dict[catid].remove(imgid)
        imgN = len(new_dict[catid])
        if imgN <imgNthreshold:
            new_dict.pop(catid)
            print('catid:%d  remain %d images, delet it!'%(catid,imgN))
        else:
            print('catid:%d  remain %d images' % (catid, imgN))
    print('remain  %d  categories'%len(new_dict))
    np.save('./utils/new_cat2imgid_dict%d.npy'%pixthreshold, new_dict)
    return new_dict

def train_data_producer(coco_item, datapath, npy, q, batch_size=10, group_size=5, img_size=224):
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.449], std=[0.226])])
    if os.path.exists(npy):
        # list_dict = np.load(npy).item()
        list_dict = np.load(npy, allow_pickle=True).item()
    else:
        list_dict = filt_small_instance(coco_item, pixthreshold=4000, imgNthreshold=100)
    catid2label={}
    n=0
    for catid in list_dict:
        catid2label[catid] = n
        n=n+1
    while 1:
        rgb = torch.zeros(batch_size*group_size, 3, img_size, img_size)
        cls_labels = torch.zeros(batch_size, 78)
        mask_labels = torch.zeros(batch_size*group_size, img_size, img_size)
        if batch_size> len(list_dict):
            remainN = batch_size - len(list_dict)
            batch_catid = random.sample(list_dict.keys(), remainN) + random.sample(list_dict, len(list_dict))
        else:
            batch_catid = random.sample(list_dict.keys(), batch_size)
        group_n = 0
        img_n = 0
        for catid in batch_catid:
            imgids = random.sample(list_dict[catid], group_size)
            co_catids = []
            anns = coco_item.imgToAnns[imgids[0]]
            for ann in anns:
                if  (ann['iscrowd'] == 0) and (ann['area'] > 4000):
                    co_catids.append(ann['category_id'])
            co_catids_backup = copy.deepcopy(co_catids)
            for imgid in imgids[1:]:
                img_catids = []
                anns = coco_item.imgToAnns[imgid]
                for ann in anns:
                    if (ann['iscrowd'] == 0) and (ann['area'] > 4000):
                        img_catids.append(ann['category_id'])
                for co_catid in co_catids_backup:
                    if co_catid not in img_catids:
                        co_catids.remove(co_catid)
                co_catids_backup = copy.deepcopy(co_catids)
            for co_catid in co_catids:
                cls_labels[group_n, catid2label[co_catid]] = 1
            for imgid in imgids:
                path = datapath + '%012d.jpg'%imgid
                img = Image.open(path)
                if img.mode == 'RGB':
                    img = img_transform(img)
                else:
                    img = img_transform_gray(img)
                anns = coco_item.imgToAnns[imgid]
                mask = None
                for ann in anns:
                    if ann['category_id'] in co_catids:
                        if mask is None:
                            mask = coco_item.annToMask(ann)
                        else:
                            mask = mask + coco_item.annToMask(ann)
                mask[mask > 0] = 255
                mask = Image.fromarray(mask)
                mask = gt_transform(mask)
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                rgb[img_n,:,:,:] = copy.deepcopy(img)
                mask_labels[img_n,:,:] = copy.deepcopy(mask)
                img_n = img_n + 1
            group_n = group_n + 1
        idx = mask_labels[:, :, :] > 1
        mask_labels[idx] = 1
        q.put([rgb, cls_labels, mask_labels])

def img_normalize(image):
    if len(image.shape)==2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel,channel,channel], axis=2)
    else:
        image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
                /np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image

davis_fbms=['bear', 'bear01', 'bear02', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'cars2', 'cars3', 'cars6', 'cars7', 'cars8', 'cars9', 'cats02', 'cats04', 'cats05', 'cats07', 'dance-jump', 'dog-agility', 'drift-turn', 'ducks01', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 'horses01', 'horses03', 'horses06', 'kite-walk', 'lion02', 'lucia', 'mallard-fly', 'mallard-water', 'marple1', 'marple10', 'marple11', 'marple13', 'marple3', 'marple5', 'marple8', 'meerkats01', 'motocross-bumps', 'motorbike', 'paragliding', 'people04', 'people05', 'rabbits01', 'rabbits05', 'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']

class VideoDataset(Dataset):
    def __init__(self, dir_,epochs, size=224, group=5, use_flow=False):
        
        self.img_list=[]
        self.label_list=[]
        self.flow_list=[]
        
        self.group=group
        
        dir_img=os.path.join(dir_,'image')
        dir_gt=os.path.join(dir_,'groundtruth')
        dir_flow=os.path.join(dir_,'flow')
        self.dir_list=sorted(os.listdir(dir_img))
        self.leng=0
        for i in range(len(self.dir_list)):
          ok=0
          if self.dir_list[i] in davis_fbms:
            ok=1
          if ok==0:
            continue
          tmp_list=[]
          cur_dir=sorted(os.listdir(os.path.join(dir_img,self.dir_list[i])))
          for j in range(len(cur_dir)):
            tmp_list.append(os.path.join(dir_img,self.dir_list[i],cur_dir[j]))
          self.leng+=len(tmp_list)
          self.img_list.append(tmp_list)
          
          tmp_list=[]
          cur_dir=sorted(os.listdir(os.path.join(dir_gt,self.dir_list[i])))
          for j in range(len(cur_dir)):
            tmp_list.append(os.path.join(dir_gt,self.dir_list[i],cur_dir[j]))
          self.label_list.append(tmp_list)
        
        self.img_size=224
        self.dataset_len = epochs
        self.use_flow=use_flow
        self.dir_=dir_
    def __len__(self):
        return self.dataset_len
        
    def __getitem__(self, item):
        
        rd=np.random.randint(0,len(self.img_list))
        rd2=np.random.permutation(len(self.img_list[rd]))
        cur_img=[]
        cur_flow=[]
        cur_gt=[]
        for i in range(self.group):
          cur_img.append(self.img_list[rd][rd2[i%len(self.img_list[rd])]])
          cur_flow.append(os.path.join(self.dir_,'flow',os.path.split(self.img_list[rd][rd2[i%len(self.img_list[rd])]])[1]))
          cur_gt.append(self.label_list[rd][rd2[i%len(self.img_list[rd])]])
          
        group_img=[]
        group_flow=[]
        group_gt=[]
        for i in range(self.group):
          tmp_img=imread(cur_img[i])
          
          tmp_img=torch.from_numpy(img_normalize(tmp_img.astype(np.float32)/255.0))
          tmp_img=F.interpolate(tmp_img.unsqueeze(0).permute(0,3,1,2),size=(self.img_size,self.img_size))
          group_img.append(tmp_img)
          
          tmp_gt=np.array(Image.open(cur_gt[i]).convert('L'))
          tmp_gt=torch.from_numpy(tmp_gt.astype(np.float32)/255.0)
          tmp_gt=F.interpolate(tmp_gt.view(1,tmp_gt.shape[0],tmp_gt.shape[1],1).permute(0,3,1,2),size=(self.img_size,self.img_size)).squeeze()
          tmp_gt=tmp_gt.view(1,tmp_gt.shape[0],tmp_gt.shape[1])
          group_gt.append(tmp_gt)
          if self.use_flow==True:
            tmp_flow=imread(cur_flow[i])
            tmp_flow=torch.from_numpy(img_normalize(tmp_flow.astype(np.float32)/255.0))
            tmp_flow=F.interpolate(tmp_flow.unsqueeze(0).permute(0,3,1,2),size=(self.img_size,self.img_size))
            group_flow.append(tmp_flow)
        
        group_img=(torch.cat(group_img,0))
        if self.use_flow==True:
          group_flow=torch.cat(group_flow,0)
        group_gt=(torch.cat(group_gt,0))
        if self.use_flow==True:
            return group_img,group_flow,group_gt
        else:
            return group_img,group_gt

