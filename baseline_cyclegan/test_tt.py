#coding:utf8
import os
import sys
import torch
import argparse
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import os

from dataset import TOTALTEXTTestloader
from metrics import runningScore
import models
import cv2
import Polygon as plg
from pse import pse
import util
import collections
from IPython import embed

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
            img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(img)
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print(img_name)
    sys.stdout.flush()
    cv2.imwrite(output_root + img_name, res)

def write_result_as_txt(image_name, bboxes, path):
    if not os.path.exists(path):
        os.makedirs(path)

    filename = util.io.join_path(path, '%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
        values = [int(v) for v in bbox]
        # line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        line = "%d"%values[0]
        for v_id in range(1, len(values)):
            line += ", %d"%values[v_id]
        line += '\n'
        lines.append(line)
    util.io.write_lines(filename, lines)

def test(args):
    n_classes = 7

    data_loader = TOTALTEXTTestloader(short_size=args.short_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=1)

     # Setup Model
    if args.arch == 'resnet18':
        model = models.resnet18(pretrained=True, num_classes=n_classes, scale=args.scale)
    elif args.arch == 'resnet18_half':
        model = models.resnet18_half(pretrained=True, num_classes=n_classes, scale=args.scale)
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=True, num_classes=n_classes, scale=args.scale)
    elif args.arch == 'resnet101':
        model = models.resnet101(pretrained=True, num_classes=n_classes, scale=args.scale)
    elif args.arch == 'resnet152':
        model = models.resnet152(pretrained=True, num_classes=n_classes, scale=args.scale)
    
    for param in model.parameters():
        param.requires_grad = False

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
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    model.eval()
    
    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()
        
        img = Variable(img.cuda(), volatile=True)
        # embed()
        #scale 0.5 2
        img = nn.functional.interpolate(img,scale_factor=0.5, mode='bilinear')
        img = nn.functional.interpolate(img, scale_factor=2.0, mode='bilinear')

        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        # print(img.shape)
        torch.cuda.synchronize()
        start = time.time()

        outputs = model(img)

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:args.kernel_num, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        
        # c++ version pse
        pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))

        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f'%(total_frame / total_time))
        sys.stdout.flush()

        label = pred
        label_num = np.max(label) + 1
        
        #方法2 先算坐标再乘scale
        scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])

        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                continue

            score_i = np.mean(score[label == i])
            if score_i < args.min_score:
                continue
            
            binary = np.zeros(label.shape, dtype='uint8')
            binary[label == i] = 1

            _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            bbox = contour

            if bbox.shape[0] <= 2:
                continue
            
            bbox = bbox * scale
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(bbox.shape[0] / 2, 2)], -1, (0, 255, 0), 2)
        
        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, bboxes, 'outputs/submit_tt/')

        # text = extend_3c(text * 255)
        # kernal = extend_3c(kernels[-1, ...] * 255)

        # org_img = cv2.resize(org_img, (text.shape[1], text.shape[0]))
        # text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        debug(idx, data_loader.img_paths, [[text_box[:,:,::-1]]], 'outputs/vis_tt/')
    cmd = 'cd eval; sh eval_tt.sh'
    os.system(cmd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='output scale')
    parser.add_argument('--short_size', nargs='?', type=int, default=800,
                        help='image short size')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.85,
                        help='min score')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    args = parser.parse_args()
    test(args)
