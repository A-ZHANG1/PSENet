# -*- coding: UTF-8 -*-
import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data

from dataset import IC15TestLoader
import models
import util
# c++ version pse based on opencv 3+ 难以编译
# from pse import pse
# python pse
from pypse import pse as pypse

def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img

def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print idx, '/', len(img_paths), img_name
    cv2.imwrite(output_root + img_name, res)

def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes=np.empty([1, 8],dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)

def test(args):
    data_loader = IC15TestLoader(long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)
    
    for param in model.parameters():
        param.requires_grad = False

    # model.cuda() will send your model to the "current device", which can be set with torch.cuda.set_device(device).
    model = model.cuda()

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model.eval()
    
    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img, s) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()

        # img = Variable(img, volatile=True)
        img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        torch.cuda.synchronize()
        start = time.time()

        outputs = model(img)

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        print("text.shape:{}".format(text.shape))
        print("outputs.shape:{}".format(outputs.shape))
        kernels = outputs[:, 0:args.kernel_num, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        
        # c++ version pse
        # pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
        # python version pse
        pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))
        
        # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
        scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
        print(pred.shape)
        print("scale:{}".format(scale))

        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        unscaled_bboxes = []
        s=s.numpy() # s: Tensor to numpy ; s.float(): 1 d tensor ro float
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                continue

            score_i = np.mean(score[label == i])
            if score_i < args.min_score:
                continue

            # print(points.shape)
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale # bbox [[a1 a2][b1 b2]...[c1 c2]]形式的nparray.可进行乘除array或单个数字的操作
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            # print(s)

            # print(cv2.boxPoints(rect))
            # print(cv2.boxPoints(rect) / s)
            unscaled_bboxes.append((cv2.boxPoints(rect) / s).astype('int32').reshape(-1))

        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f'%(total_frame / total_time))
        sys.stdout.flush()


        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        # 1 检测框txt路径 dataset_model
        # write_result_as_txt(image_name, bboxes, 'outputs/submit_funs_model3_epoch_48/')
        write_result_as_txt(image_name, bboxes, 'test_img/')
        #write_result_as_txt(image_name, bboxes, 'outputs/submit_funs_model1/')
        # 2 unscaled txt 路径
        # write_result_as_txt(image_name, unscaled_bboxes, 'outputs/submit_funs_model3_epoch_48/')

        
        text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        print("text_box.shape:{}".format(text_box.shape))
        # 3 img 路径
        # debug(idx, data_loader.img_paths, [[text_box]], 'outputs/vis_ic15/')
        # debug(idx, data_loader.img_paths, [[text_box]], 'outputs/vis_funs_model3_epoch_48/')
        debug(idx, data_loader.img_paths, [[text_box]], 'test_img/')

    # 4 submit.zip制作
    # cmd = 'cd %s;zip -j %s %s/*'%('./outputs/', 'submit_ic15.zip', 'submit_ic15');
    # cmd = 'cd %s;zip -j %s %s/*'%('./outputs/', 'submit_funs_model3_epoch_48.zip', 'submit_funs_model3_epoch_48');
    sys.stdout.flush()
    # util.cmd.cmd(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    # 5 model1/model2
    parser.add_argument('--resume', nargs='?', type=str, default='ic15_res50_pretrain_ic17.pth.tar',
    # parser.add_argument('--resume', nargs='?', type=str, default='checkpoints/model_funs_pretrain_ic15_frozen_res50_layer1_pretrain_ic17/checkpoint_epoch48.pth.tar',
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=2240,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
    
    args = parser.parse_args()
    test(args)
