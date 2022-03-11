import os
from PIL import Image
import torch
from torchvision import transforms
from model_video import build_model
import numpy as np
import cv2
import argparse
import imageio as ig
import moviepy.editor as mp
to_pil = transforms.ToPILImage()

def main(gpu_id, model_path, datapath, save_root_path, group_size, img_size, img_dir_name,crf):
    net = build_model(device).to(device)
    net=torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path, map_location=gpu_id))
    net.eval()
    net = net.module.to(device)
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
                                             
    
    with torch.no_grad():
        for p in range(len(datapath)):
            frame_dir=os.path.join(os.path.split(os.path.split(datapath[p])[0])[0],'frame')
            result_dir=os.path.join(os.path.join(os.path.split(os.path.split(datapath[p])[0])[0],'result'),os.path.split(datapath[p])[-1])
            
            
            vc = cv2.VideoCapture(datapath[p])
            rval = vc.isOpened()
            c=0
            frame_list=[]
            frame_name_list=[]
            while rval:
                rval, frame = vc.read()
                if rval:
                    if(c>9999):
                        break
                    if(c//10==0):
                        frame_name_list.append(os.path.join(frame_dir,"000"+str(c) + '.jpg'))
                        cv2.imwrite(os.path.join(frame_dir,"000"+str(c) + '.jpg'), frame) #000i
                        frame_list.append(frame)
                    elif(c//100==0):
                        frame_name_list.append(os.path.join(frame_dir,"00"+str(c) + '.jpg'))
                        cv2.imwrite(os.path.join(frame_dir,"00"+str(c) + '.jpg'), frame) #00i
                        frame_list.append(frame)
                    elif(c//1000==0):
                        frame_name_list.append(os.path.join(frame_dir,"0"+str(c) + '.jpg'))
                        cv2.imwrite(os.path.join(frame_dir,"0"+str(c) + '.jpg'), frame) #0i
                        frame_list.append(frame)
                    else:
                        frame_name_list.append(os.path.join(frame_dir,str(c) + '.jpg'))
                        cv2.imwrite(os.path.join(frame_dir,str(c) + '.jpg'), frame) #i
                        frame_list.append(frame)
                    c=c+1
                else:
                    break
            vc.release()
            #frame_name_list=frame_name_list[7000:]
            #frame_list=frame_list[7000:]
            all_class = ['frame']
            image_list, save_list = list(), list()
            for s in range(len(all_class)):
                image_path = frame_name_list #sorted(os.listdir(os.path.join(datapath[p], img_dir_name, all_class[s])))
                idx=[]
                block_size=(len(image_path)+group_size-1)//group_size
                for ii in range(block_size):
                  cur=ii
                  while cur<len(image_path):
                    idx.append(cur)
                    cur+=block_size
                new_image_path=[]
                for ii in range(len(image_path)):
                  new_image_path.append(image_path[idx[ii]])
                image_path=new_image_path
                if(len(image_path)<=2): #wrong directory
                  continue
                image_list.append(image_path ) #list(map(lambda x: os.path.join(datapath[p], img_dir_name, all_class[s], x), image_path)))
                save_list.append(image_path) #list(map(lambda x: os.path.join(save_root_path[p], all_class[s], x[:-4]+'.jpg'), image_path)))
            
            frame_result=[]
            for i in range(len(image_list)):
                cur_class_all_image = image_list[i]
                cur_class_rgb = torch.zeros(len(cur_class_all_image), 3, img_size, img_size)
                original_img=[]
                for m in range(len(cur_class_all_image)):
                    rgb_ = Image.open(cur_class_all_image[m])
                    if rgb_.mode == 'RGB':
                        rgb_ = img_transform(rgb_)
                    else:
                        rgb_ = img_transform_gray(rgb_)
                    original_img.append(Image.open(cur_class_all_image[m]))
                    cur_class_rgb[m, :, :, :] = rgb_

                cur_class_mask = torch.zeros(len(cur_class_all_image), img_size, img_size)
                divided = len(cur_class_all_image) // group_size
                rested = len(cur_class_all_image) % group_size
                if divided != 0:
                    for k in range(divided):
                        group_rgb = cur_class_rgb[(k * group_size): ((k + 1) * group_size)]
                        group_rgb = group_rgb.to(device)
                        _, pred_mask = net(group_rgb)
                        cur_class_mask[(k * group_size): ((k + 1) * group_size)] = pred_mask
                if rested != 0:
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[:group_size - rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    group_rgb = group_rgb.to(device)
                    _, pred_mask = net(group_rgb)
                    cur_class_mask[(divided * group_size):] = pred_mask[:rested]


                class_save_path = os.path.join(save_root_path[p], all_class[i])
                if not os.path.exists(class_save_path):
                    os.makedirs(class_save_path)

                for j in range(len(cur_class_all_image)):
                    exact_save_path = save_list[i][j]
                    result = cur_class_mask[j, :, :]
                    prediction = np.array(to_pil(result.data.squeeze().cpu()))
                    
                    #result = Image.fromarray(result * 255)
                    img=Image.open(image_list[i][j])
                    w, h = img.size
                    img=img.resize((prediction.shape[0],prediction.shape[1]),Image.BILINEAR)
                    if crf==True:
                        prediction = crf_refine(np.array(img), prediction)
                    result=torch.from_numpy(np.array(prediction)/255).view(prediction.shape[0],prediction.shape[1],1).repeat(1,1,3).numpy()
                    img=np.array(img)
                    result=(img/2+np.array([127,127,0]))*result+(1-result)*img
                    #print(type(result))
                    result=Image.fromarray(result.astype(np.uint8))
                    result = result.resize((w, h), Image.BILINEAR)
                    #result.save(exact_save_path)
                    frame_result.append(result)
            new_frame_result=[]
            for index in range(len(frame_result)):
              new_frame_result.append(frame_result[index])
            for index in range(len(frame_result)):
              new_frame_result[idx[index]]=frame_result[index]
            
            order = 0
            name=os.path.join(os.path.split(os.path.split(datapath[p])[0])[0],'temp')
            frames=[]
            for img in new_frame_result:
              frames.append(np.array(img))
              
            ig.mimsave(name, frames, 'GIF', duration=0.05)
            clip = mp.VideoFileClip(name)
            clip.write_videofile(result_dir + '.mp4')
            os.remove(name)
            for f in frame_name_list:
              os.remove(f)
            print('done')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/video_best.pth', help="restore checkpoint")
    parser.add_argument('--data_path',default='./demo_mp4/video/kobe_1v1.mp4', help="dataset for evaluation")
    parser.add_argument('--output_dir', default='./demo_mp4/result', help='directory for result')
    parser.add_argument('--gpu_id', default='cuda:0', help='id of gpu')
    parser.add_argument('--crf', default=False, help='make outline clear')
    args = parser.parse_args()
    
    gpu_id = args.gpu_id
    device = torch.device(gpu_id)
    model_path = args.model

    val_datapath = [args.data_path]

    save_root_path = [args.output_dir]

    main(gpu_id, model_path, val_datapath, save_root_path, 5, 224, 'image',args.crf)
