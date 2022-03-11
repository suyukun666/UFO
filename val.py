from torchvision import transforms
import torch
import os
from PIL import Image
from tools import *
import argparse
import datetime


def validation(net, datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
               img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png']):
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    net.eval()
    net = net.module.to(device)
    with torch.no_grad():
        ave_p, ave_j = [], []
        for p in range(len(datapath)):
            all_p, all_j = [], []
            all_class = os.listdir(os.path.join(datapath[p], img_dir_name))
            image_list, gt_list = list(), list()
            for s in range(len(all_class)):
                image_path = os.listdir(os.path.join(datapath[p], img_dir_name, all_class[s]))
                image_list.append(list(map(lambda x: os.path.join(datapath[p], img_dir_name, all_class[s], x), image_path)))
                gt_list.append(list(map(lambda x: os.path.join(datapath[p], gt_dir_name, all_class[s], x.replace(img_ext[p], gt_ext[p])), image_path)))
            for i in range(len(image_list)):
                cur_class_all_image = image_list[i]
                cur_class_all_gt = gt_list[i]

                cur_class_gt = torch.zeros(len(cur_class_all_gt), img_size, img_size)
                for g in range(len(cur_class_all_gt)):
                    gt_ = Image.open(cur_class_all_gt[g]).convert('L')
                    gt_ = gt_transform(gt_)
                    gt_[gt_ > 0.5] = 1
                    gt_[gt_ <= 0.5] = 0
                    cur_class_gt[g, :, :] = gt_

                cur_class_rgb = torch.zeros(len(cur_class_all_image), 3, img_size, img_size)
                for m in range(len(cur_class_all_image)):
                    rgb_ = Image.open(cur_class_all_image[m])
                    if rgb_.mode == 'RGB':
                        rgb_ = img_transform(rgb_)
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
                        _, pred_mask = net(group_rgb)
                        
                        cur_class_mask[(k * group_size): ((k + 1) * group_size)] = pred_mask
                if rested != 0:
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[:group_size-rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    # group_rgb = group_rgb.to(device)
                    group_rgb = group_rgb.cuda()
                    _, pred_mask = net(group_rgb)
                    cur_class_mask[(divided * group_size): ] = pred_mask[:rested]

                for q in range(cur_class_mask.size(0)):
                    single_p, single_j = calc_precision_and_jaccard(cur_class_mask[q, :, :].numpy(), cur_class_gt[q, :, :].numpy())
                    all_p.append(single_p)
                    all_j.append(single_j)

            dataset_p = np.mean(all_p)
            dataset_j = np.mean(all_j)

            ave_p.append(dataset_p)
            ave_j.append(dataset_j)

    return ave_p, ave_j

def validation_with_flow(net, datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
               img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png']):
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    net.eval()
    net = net.module.to(device)
    with torch.no_grad():
        ave_p, ave_j = [], []
        for p in range(len(datapath)):
            all_p, all_j = [], []
            all_class = sorted(os.listdir(os.path.join(datapath[p], img_dir_name)))
            image_list,flow_list, gt_list = list(), list(),list()
            for s in range(len(all_class)):
                image_path = sorted(os.listdir(os.path.join(datapath[p], img_dir_name, all_class[s])))
                flow_path = sorted(os.listdir(os.path.join(datapath[p], 'flow', all_class[s])))
                if len(image_path)>len(flow_path):
                  image_path=image_path[:-1]
                
                image_list.append(sorted(list(map(lambda x: os.path.join(datapath[p], img_dir_name, all_class[s], x), image_path))))
                flow_list.append(sorted(list(map(lambda x: os.path.join(datapath[p], 'flow', all_class[s], x.replace(img_ext[p], '.jpg')), flow_path))))
                gt_list.append(sorted(list(map(lambda x: os.path.join(datapath[p], gt_dir_name, all_class[s], x.replace(img_ext[p], gt_ext[p])), image_path))))
                
            for i in range(len(image_list)):
                cur_class_all_image = image_list[i]
                cur_class_all_gt = gt_list[i]
                cur_class_all_flow=flow_list[i]
                
                cur_class_gt = torch.zeros(len(cur_class_all_gt), img_size, img_size)
                for g in range(len(cur_class_all_gt)):
                    gt_ = Image.open(cur_class_all_gt[g]).convert('L')
                    gt_ = gt_transform(gt_)
                    gt_[gt_ > 0.5] = 1
                    gt_[gt_ <= 0.5] = 0
                    cur_class_gt[g, :, :] = gt_

                cur_class_rgb = torch.zeros(len(cur_class_all_image), 3, img_size, img_size)
                cur_class_flow=torch.zeros(len(cur_class_all_flow),3,img_size,img_size)
                
                for m in range(len(cur_class_all_flow)):
                  flow_=Image.open(cur_class_all_flow[m])
                  if flow_.mode=='RGB':
                    flow_=img_transform(flow_)
                  else:
                    flow_=img_transform_gray(rgb_)
                  cur_class_flow[m,:,:,:]=flow_
                  
                
                for m in range(len(cur_class_all_image)):
                    rgb_ = Image.open(cur_class_all_image[m])
                    if rgb_.mode == 'RGB':
                        rgb_ = img_transform(rgb_)
                    else:
                        rgb_ = img_transform_gray(rgb_)
                    cur_class_rgb[m, :, :, :] = rgb_

                cur_class_mask = torch.zeros(len(cur_class_all_image), img_size, img_size)
                divided = len(cur_class_all_image) // group_size
                rested = len(cur_class_all_image) % group_size
                if divided != 0:
                    for k in range(divided):
                        group_rgb = cur_class_rgb[(k * group_size): ((k + 1) * group_size)]
                        group_flow=cur_class_flow[(k * group_size): ((k + 1) * group_size)].cuda()
                        # group_rgb = group_rgb.to(device)
                        group_rgb = group_rgb.cuda()
                        _,pred_mask = net(group_rgb,group_flow)
                        
                        cur_class_mask[(k * group_size): ((k + 1) * group_size)] = pred_mask
                if rested != 0:
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[:group_size-rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    
                    group_flow_tmp_l = cur_class_flow[-rested:]
                    group_flow_tmp_r = cur_class_flow[:group_size-rested]
                    group_flow = torch.cat((group_flow_tmp_l, group_flow_tmp_r), dim=0).cuda()
                    # group_rgb = group_rgb.to(device)
                    group_rgb = group_rgb.cuda()
                    _,pred_mask = net(group_rgb,group_flow)
                    cur_class_mask[(divided * group_size): ] = pred_mask[:rested]

                for q in range(cur_class_mask.size(0)):
                    single_p, single_j = calc_precision_and_jaccard(cur_class_mask[q, :, :].numpy(), cur_class_gt[q, :, :].numpy())
                    all_p.append(single_p)
                    all_j.append(single_j)

            dataset_p = np.mean(all_p)
            dataset_j = np.mean(all_j)

            ave_p.append(dataset_p)
            ave_j.append(dataset_j)

    return ave_p, ave_j
    
if __name__=='__main__':
    from model_video import build_model, weights_init
    parser = argparse.ArgumentParser()
    parser.add_argument('--vgg16_path', default='./weights/vgg16_bn_feat.pth',help="vgg path")
    parser.add_argument('--npy_path',default='./utils/new_cat2imgid_dict4000.npy', help="npy path")
    parser.add_argument('--output_dir', default='./VSOD_results/wo_optical_flow/DAVIS/', help='directory for result')
    parser.add_argument('--task', default='CoS', choices=['CoS','VSOD','CoSD'],help='task')
    parser.add_argument('--use_flow', default=False, help='use flow or not')
    parser.add_argument('--gpu_id', default='cuda:0', help='id of gpu')
    parser.add_argument('--crf', default=False, help='make outline clear')
    parser.add_argument('--img_size', default=224, help='image size')
    parser.add_argument('--lr', default=1e-5, help='learning rate')
    parser.add_argument('--lr_de', default=20000, help='learning rate decay')
    parser.add_argument('--epochs', default=100000, help='epochs')
    parser.add_argument('--bs', default=8, help='batch size')
    parser.add_argument('--gs', default=5, help='group size')
    parser.add_argument('--log_interval', default=100, help='log interval')
    parser.add_argument('--val_interval', default=1000, help='val interval')
    parser.add_argument('--model', default='./models/video_best.pth',help="restore checkpoint")
    args = parser.parse_args()
    
    val_datapath = ['./cosegdatasets/DAVIS',
                    './cosegdatasets/MSRC7',
                    './cosegdatasets/Internet_Datasets300',
                    './cosegdatasets/PASCAL_VOC']
    device='cuda:0'
    device = torch.device('cuda:0')
    img_size = args.img_size
    lr = args.lr
    lr_de = args.lr_de
    epochs = args.epochs
    batch_size = args.bs
    group_size = args.gs
    log_interval = args.log_interval
    val_interval = args.val_interval
    gpu_id='cuda:0'
    net = build_model(device).to(device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    state_dict = torch.load(args.model, map_location=gpu_id)
    net.load_state_dict(state_dict)
    
    ave_p, ave_j = validation(net, val_datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
                                          img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png'])
    #custom_print('-' * 100, log_txt_file, 'a+')
    print(datetime.datetime.now().strftime('%F %T') + ' iCoseg8  p: [%.4f], j: [%.4f]' %
                             (ave_p[0], ave_j[0]))
    print(datetime.datetime.now().strftime('%F %T') + ' MSRC7    p: [%.4f], j: [%.4f]' %
                             (ave_p[1], ave_j[1]))
    print(datetime.datetime.now().strftime('%F %T') + ' Int_300  p: [%.4f], j: [%.4f]' %
                             (ave_p[2], ave_j[2]))
    print(datetime.datetime.now().strftime('%F %T') + ' PAS_VOC  p: [%.4f], j: [%.4f]' %
                             (ave_p[3], ave_j[3]))
    #custom_print('-' * 100, log_txt_file, 'a+')