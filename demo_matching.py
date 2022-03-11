import os
import torch
import argparse
import queue
import threading
from model_image import build_model, weights_init
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import time
import datetime

import torch.nn.functional as F
torch.backends.cudnn.benchmark = True


def Idx(cur,ii,jj):
  return ii*14+jj+cur*14*14

def pix_idx(ii,jj):
  return (ii*16+8,jj*16+8)

def main(net, datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
               img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png'],output_dir='./matching_result'):
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    res_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    net.eval()
    net = net.module.to(device)
    
    col_tab=[(101,67,254),(154,157,252),(173,205,249),(169,200,200),(155,175,131)]
    with torch.no_grad():
        ave_p, ave_j = [], []
        for p in range(len(datapath)):
            
            all_p, all_j = [], []
            all_class = [os.path.split(datapath[p])[-1]]
            
            datapath[p]=os.path.split(os.path.split(datapath[p])[0])[0]
            cur_idx=0
            image_list, gt_list = list(), list()
            for s in range(len(all_class)):
                image_path = sorted(os.listdir(os.path.join(datapath[p], img_dir_name, all_class[s])))
                image_list.append(list(map(lambda x: os.path.join(datapath[p], img_dir_name, all_class[s], x), image_path)))
                gt_list.append(list(map(lambda x: os.path.join(datapath[p], gt_dir_name, all_class[s], x.replace(img_ext[p], gt_ext[p])), image_path)))
            for i in range(len(image_list)):
                cur_class_all_image = sorted(image_list[i])
                cur_class_all_gt = gt_list[i]

                cur_class_gt = torch.zeros(len(cur_class_all_gt), img_size, img_size)
                
                cur_class_rgb = torch.zeros(len(cur_class_all_image), 3, img_size, img_size)
                real_img=[]
                
                idx=0
                idx_i,idx_j=3,4
                for m in range(len(cur_class_all_image)):
                    rgb_ = Image.open(cur_class_all_image[m])
                    if rgb_.mode == 'RGB':
                        rgb_ = img_transform(rgb_)
                        ans_ori=cv2.cvtColor((res_transform(Image.open(cur_class_all_image[m]).convert('RGB'))*255).permute(1,2,0).numpy().astype(np.uint8),cv2.COLOR_BGR2RGB)
                        real_img.append(ans_ori*1)
                        
                    else:
                        rgb_ = img_transform_gray(rgb_)
                    cur_class_rgb[m, :, :, :] = rgb_

                cur_class_mask = torch.zeros(len(cur_class_all_image), img_size, img_size)
                divided = len(cur_class_all_image) // group_size
                rested = len(cur_class_all_image) % group_size
                if divided != 0:
                    for k in range(divided):
                        group_rgb = cur_class_rgb[(k * group_size): ((k + 1) * group_size)]
                        # group_rgb = group_rgb.to(device)
                        group_rgb = group_rgb.cuda()
                        _, pred_mask,feat,feat2 = net(group_rgb)
                        feat_list=[]
                        #print(feat.shape)
                        first=None
                        for j in range(group_size):
                          x_visualize = feat[j].unsqueeze(0).cpu() 
                          
                          x_visualize = -np.mean(x_visualize.numpy(),axis=1).reshape(x_visualize.shape[-2],x_visualize.shape[-1])
                          
                          x_visualize=(x_visualize-x_visualize.min())/(x_visualize.max()-x_visualize.min())
                          feat_list.append(x_visualize)
                          CAM = cv2.applyColorMap((x_visualize*255).astype(np.uint8), cv2.COLORMAP_JET)
                          CAM=F.interpolate(torch.from_numpy(CAM).permute(2,0,1).float().view(1,3,CAM.shape[0],CAM.shape[1]),size=[224,224],mode='bilinear').squeeze().permute(1,2,0).numpy().astype(np.uint8)
                          cv2.imwrite(os.path.join(output_dir,str(j)+'_CAM.png'),CAM)
                          if first is None:
                            first=x_visualize
                          #break
                        th=0.9
                        cat_img=np.concatenate([real_img[0],real_img[1]],axis=1)
                        cur_patch=0
                        for ii in range(14):
                          for jj in range(14):
                            if first[ii][jj]>=th:
                              
                              Max=-10
                              Max_pp=0
                              Max_qq=0
                              for pp in range(14):
                                for qq in range(14):
                                  if feat2[Idx(0,ii,jj)][Idx(1,pp,qq)]>Max:
                                    Max_pp=pp
                                    Max_qq=qq
                                    Max=feat2[Idx(0,ii,jj)][Idx(1,pp,qq)]
                              cur_patch+=1
                              
                              cv2.line(cat_img,pix_idx(jj,ii),pix_idx(Max_qq+14,Max_pp),col_tab[cur_patch%len(col_tab)],1)
                              cv2.circle(cat_img,pix_idx(jj,ii),1,col_tab[cur_patch%len(col_tab)],1)
                              cv2.circle(cat_img,pix_idx(Max_qq+14,Max_pp),1,col_tab[cur_patch%len(col_tab)],1)
                              
                        cur_class_mask[(k * group_size): ((k + 1) * group_size)] = pred_mask
                        cv2.imwrite('./matching_result/matching_result.jpg',cat_img)
                if rested != 0:
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[:group_size-rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    group_rgb = group_rgb.cuda()
                    _, pred_mask,feat = net(group_rgb)
                    for j in range(group_size):
                      x_visualize = feat[j].unsqueeze(0).cpu().numpy()
                      x_visualize = np.mean(x_visualize,axis=1).reshape(x_visualize.shape[-2],x_visualize.shape[-1])
                      x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8)
                      savedir =  './visual_of_transformer/'        
                    
                      x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET) 
                      ans=torch.zeros(img_size,img_size,3).numpy()
                      for ii in range(img_size):
                          for jj in range(img_size):
                              ans[ii][jj]=x_visualize[ii//((img_size+group_size-1)//feat_size)][jj//((img_size+group_size-1)//feat_size)]
                      cur_idx+=1 
                      cv2.imwrite(savedir+str(cur_idx)+'.jpg',ans)
                    cur_class_mask[(divided * group_size): ] = pred_mask[:rested]

if __name__ == '__main__':
    # train_val_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/image_best.pth',help="restore checkpoint")
    parser.add_argument('--data_path',default='./matching_data/image/camel', help="dataset for evaluation")
    parser.add_argument('--output_dir',default='./matching_result', help="dataset for evaluation")
    args = parser.parse_args()
    
    val_datapath = [args.data_path]
                                        
    # project config
    project_name = 'UFO'
    device = torch.device('cuda:0')
    img_size = 224
    lr = 1e-5
    lr_de = 20000
    epochs = 100000
    batch_size = 4
    group_size = 5
    log_interval = 100
    val_interval = 1000

    model_path = args.model
    gpu_id='cuda:0'
    device = torch.device(gpu_id)
    net = build_model(device,demo_mode=True).to(device)
    net=torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, map_location=gpu_id))
    
    net.eval()
    with torch.no_grad():         
                main(net, val_datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
                                          img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png'],output_dir=args.output_dir)