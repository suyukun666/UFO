#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import torch


#read video
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',default='add_bullet_chat_to_video/cat/image', help="dataset")
parser.add_argument('--choice', default='0',choices=['0','1'], help='0 denotes that you use this code to get frame from video and 1 denotes that you use this code to create a demo video')
parser.add_argument('--video_path',default='add_bullet_chat_to_video/cat.mp4', help="video path")
parser.add_argument('--output_path',default='add_bullet_chat_to_video/demo-result.mp4', help="output path")
args = parser.parse_args()

result_video = args.output_path
video = args.video_path
cap = cv2.VideoCapture(video)

#get FPS
fps_video = cap.get(cv2.CAP_PROP_FPS)
#bianmageshi
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#width
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
videoWriter = cv2.VideoWriter(result_video, fourcc, fps_video, (frame_width*3, frame_height))
frame_id = 0

# you can use your own text
danmu=['Great job','Oh','Favorable background'
      ,'Respect','never get tired of looking at it','Every frame is a classic','Hi','great','I like it','Oh',
      'This video is great','Favorable background','Friends, we meet again','see you at the beginning',
      'Nice work','Oh','Favorable background'
      ,'Respect','never get tired of looking at it','Every frame is a classic','Hi','great','I like it','Oh',
      'This video is great','Favorable background','Friends, we meet again','see you at the beginning',
      'Good job','Oh','Favorable background'
      ,'Respect','never get tired of looking at it','Every frame is a classic','Hi','great','I like it','Oh',
      'This video is great','Favorable background','Friends, we meet again','see you at the beginning',
      'oh!!!!!','Oh','Favorable background'
      ,'Respect','never get tired of looking at it','Every frame is a classic','Hi','great','I like it','Oh',
      'it is great','Favorable background','Friends, we meet again','see you at the beginning of the video'
      ,'good night','good morning','hello','Nice to meet you here','Anyone here?','Respect','never get tired of looking at it',
      'Every frame is a classic','Hi','great','I like it','Oh',
      'This video is great','Favorable background','Friends, we meet again','see you at the beginning of the video'
      ,'good night','good morning','hello','Nice to meet you here','Anyone here?']

y_=[]
x_=[]
for i in range(len(danmu)):
  if i%10<5:
    y_.append(i%10*40+70)
  else:
    y_.append((i%10)*40+70)
for i in range(len(danmu)):
  x_.append(frame_height+i*8)
  x_[len(x_)-1]+=i//10*300

speed=3
tot=0
cur=args.data_path
lis=sorted(os.listdir(cur))
color_tab=[(101,67,254),(154,157,252),(173,205,249),(169,200,200),(155,175,131),(148,137,69),(185,235,35)]
while (cap.isOpened()):
    
    ret, frame = cap.read()
    if ret == True:
        
        frame_id += 1
        if tot>=10000:
          break
        left_x_up = int(frame_width / frame_id)
        left_y_up = int(frame_height / frame_id)
        '''right_x_down = int(left_x_up + frame_width / 10)
        right_y_down = int(left_y_up + frame_height / 10)'''
        #coordinate
        word_x = left_x_up + 5
        word_y = left_y_up + 25
        
        original=frame*1
        
        
        if args.choice=='1':
            for i in range(len(danmu)): 
              word_x=x_[i]-tot*speed
              word_y=y_[i]
              cv2.putText(frame, danmu[i], (word_x, word_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color_tab[i%len(color_tab)], 2)
            covered=frame*1
            mask=cv2.imread(os.path.join(cur,lis[tot]))
            os.remove(os.path.join(cur,lis[tot]))
            os.remove(os.path.join(os.path.join(cur,lis[tot]).split('/')[0],os.path.join(cur,lis[tot]).split('/')[1],'image',os.path.join(cur,lis[tot]).split('/')[-2],os.path.join(cur,lis[tot]).split('/')[-1]))
            mask=((torch.from_numpy(mask)/255)>0.3).float().numpy()
            not_covered=mask*original+(1-mask)*covered
            
            frame=np.concatenate([original,covered,not_covered],axis=1)
            videoWriter.write(frame.astype(np.uint8))
            tot+=1
            continue
        cate=os.path.split(os.path.split(cur)[0])[-1]
        rt=os.path.split(args.video_path)[0]
        if not os.path.exists(os.path.join(rt,cate+'/image/'+cate)):
          os.mkdir(os.path.join(rt,cate+'/image/'+cate))
        if tot<10:
          cv2.imwrite(os.path.join(rt,cate+'/image/'+cate+'/'+'000'+str(tot)+'.png'),frame)
        elif tot<100:
          cv2.imwrite(os.path.join(rt,cate+'/image/'+cate+'/'+'00'+str(tot)+'.png'),frame)
        elif tot<1000:
          cv2.imwrite(os.path.join(rt,cate+'/image/'+cate+'/'+'0'+str(tot)+'.png'),frame)
        else:
          cv2.imwrite(os.path.join(rt,cate+'/image/'+cate+'/'+str(tot)+'.png'),frame)
        tot+=1
    else:
        videoWriter.release()
        break
#print(tot)