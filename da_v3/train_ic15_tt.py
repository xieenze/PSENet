#coding:utf8
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torch.utils import data
import os

from dataset import IC15_TT_Loader
from metrics import runningScore
import models
from util import Logger, AverageMeter
import time
import util

from loss_opr import *
import loss_opr

from IPython import embed


def get_G_LOSS(outputs, criterion, gt_kernels, gt_texts, training_masks):
    texts = outputs[:, 0, :, :]
    kernels = outputs[:, 1:, :, :]
    selected_masks = ohem_batch(texts, gt_texts, training_masks)
    loss_text = criterion(texts, gt_texts, selected_masks)
    loss_kernels = []
    mask0 = torch.sigmoid(texts)
    mask1 = training_masks
    selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).float()
    for i in range(6):
        kernel_i = kernels[:, i, :, :]
        gt_kernel_i = gt_kernels[:, i, :, :]
        loss_kernel_i = criterion(kernel_i, gt_kernel_i, selected_masks)
        loss_kernels.append(loss_kernel_i)
    loss_kernel = sum(loss_kernels) / len(loss_kernels)

    loss_kernel = loss_kernel * 0.7
    loss_text = loss_text * 0.3
    loss = loss_text + loss_kernel
    return loss, loss_kernel, loss_text


def train(train_loader, model_G, model_D, criterion, optimizer_G, optimizer_D, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)


    # labels for adversarial training
    source_label = 0
    target_label = 1
    bce_loss = torch.nn.CrossEntropyLoss()

    end = time.time()
    for batch_idx, (source_imgs, gt_texts, gt_kernels, training_masks, target_imgs) in enumerate(train_loader):
        data_time.update(time.time() - end)

        source_imgs = Variable(source_imgs.cuda())
        target_imgs = Variable(target_imgs.cuda())
        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())
        training_masks = Variable(training_masks.cuda())


        # train G
        # don't accumulate grads in D

        model_G.zero_grad()

        for param in model_D.parameters():
            param.requires_grad = False
        # for param in model_G.parameters():
        #     param.requires_grad = True

        # train with source
        outputs_source_fmap, outputs_source = model_G(source_imgs)
        loss, loss_kernel, loss_text = get_G_LOSS(outputs_source, criterion, gt_kernels, gt_texts, training_masks)
        loss.backward(retain_graph=True)

        optimizer_G.step()

        model_G.zero_grad()

        D_out1 = model_D(outputs_source_fmap)
        loss_adv_s = bce_loss(D_out1, Variable(torch.LongTensor(D_out1.size()[0]).fill_(target_label)).cuda())

        # train with target
        outputs_target_fmap, outputs_target = model_G(target_imgs)
        D_out1 = model_D(outputs_target_fmap)
        loss_adv_t = bce_loss(D_out1, Variable(torch.LongTensor(D_out1.size()[0]).fill_(source_label)).cuda())

        loss_adv = 0.01 * loss_adv_s + 0.01 * loss_adv_t
        loss_adv.backward()

        optimizer_G.step()




        # train D
        # bring back requires_grad

        model_D.zero_grad()

        for param in model_D.parameters():
            param.requires_grad = True
        # don't accumulate grads in G
        # for param in model_G.parameters():
        #     param.requires_grad = False

        # train with source
        outputs_source_fmap = outputs_source_fmap.detach()
        D_out1 = model_D(outputs_source_fmap)
        loss_D1 = bce_loss(D_out1, Variable(torch.LongTensor(D_out1.size()[0]).fill_(source_label)).cuda())
        # loss_D1.backward()

        # train with target
        outputs_target_fmap = outputs_target_fmap.detach()
        D_out1 = model_D(outputs_target_fmap)
        loss_D2 = bce_loss(D_out1, Variable(torch.LongTensor(D_out1.size()[0]).fill_(target_label)).cuda())
        # loss_D2.backward()
        loss_D = loss_D1 + loss_D2
        loss_D.backward()

        optimizer_D.step()



        batch_time.update(time.time() - end)
        end = time.time()


        if batch_idx % 20 == 0:
            texts = outputs_source[:, 0, :, :]
            kernels = outputs_source[:, 1:, :, :]
            score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
            score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)
            output_log  = 'Epoch:{epoch} | Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.3f} | Loss_K: {loss_kernel:.3f} | Loss_T: {loss_text:.3f} | Loss_adv: {loss_adv:.3f} | Loss_D1: {loss_d1:.3f} | Loss_D2: {loss_d2:.3f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f} | IOU_k: {iou_k: .4f}'.format(
                epoch=epoch,
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=loss,
                loss_kernel=loss_kernel,
                loss_text=loss_text,
                loss_adv=loss_adv,
                loss_d1=loss_D1,
                loss_d2=loss_D2,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'],
                iou_k=score_kernel['Mean IoU'])
            print(output_log)
            sys.stdout.flush()
    return output_log

def adjust_learning_rate_G(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr_G = args.lr_G * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_G

def adjust_learning_rate_D(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr_D = args.lr_D * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_D

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main(args):
    # 据说能加速
    cudnn.enabled = True
    cudnn.benchmark = True
    args.batch_size *= torch.cuda.device_count()
    print("batch size=",args.batch_size)

    if args.checkpoint == '':
        args.checkpoint = "checkpoints/ic15_%s_bs_%d_ep_%d"%(args.arch, args.batch_size, args.n_epoch)
    if args.pretrain:
        if 'synth' in args.pretrain:
            args.checkpoint += "_pretrain_synth"
        else:
            args.checkpoint += "_pretrain_ic17"

    print ('\ncheckpoint path: %s'%args.checkpoint)
    print ('init lr: %.8f'%args.lr_G)
    print ('schedule: ', args.schedule)
    print('loading model.....')
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    kernel_num = 7
    min_scale = 0.4
    start_epoch = 0

    data_loader = IC15_TT_Loader(is_transform=True, img_size=args.img_size, kernel_num=kernel_num, min_scale=min_scale)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=20,
        drop_last=True,
        pin_memory=True)

    #加载生成器G
    model_G = models.resnet50(pretrained=True, num_classes=kernel_num)
    model_G = torch.nn.DataParallel(model_G).cuda().train()

    #加载判别网络D
    model_D = models.FCDiscriminator(num_classes=256)
    model_D = torch.nn.DataParallel(model_D).cuda().train()

    optimizer_G  = torch.optim.SGD(model_G.parameters(), lr=args.lr_G, momentum=0.99, weight_decay=5e-4)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr_D, betas=(0.9, 0.99))

    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model_G.load_state_dict(checkpoint['state_dict'])
    elif args.resume:
        print('Resuming from checkpoint.')
        #load G
        G_path = os.path.join(args.resume,"G.pth")
        checkpoint_G = torch.load(G_path)
        start_epoch = checkpoint_G['epoch']
        model_G.load_state_dict(checkpoint_G['state_dict'])
        optimizer_G.load_state_dict(checkpoint_G['optimizer'])
        #load D
        D_path = os.path.join(args.resume, "D.pth")
        checkpoint_D = torch.load(D_path)
        model_D.load_state_dict(checkpoint_D['state_dict'])
        optimizer_D.load_state_dict(checkpoint_D['optimizer'])
    else:
        print('Training from scratch.')



    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate_G(args, optimizer_G, epoch)
        adjust_learning_rate_D(args, optimizer_D, epoch)

        print('\nEpoch: [{} | {}] LR_G: {}  LR_D: {}'.format(epoch + 1, args.n_epoch,
                                                             optimizer_G.param_groups[0]['lr'],
                                                             optimizer_D.param_groups[0]['lr']))

        # start training.....
        output_log = train(train_loader,
                      model_G, model_D,
                      dice_loss,
                      optimizer_G, optimizer_D,
                      epoch)


        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_G.state_dict(),
                'lr': args.lr_G,
                'optimizer' : optimizer_G.state_dict(),
            }, checkpoint=args.checkpoint,
                filename='G.pth')

        save_checkpoint({
            'state_dict': model_D.state_dict(),
            'lr': args.lr_D,
            'optimizer': optimizer_D.state_dict(),
            }, checkpoint=args.checkpoint,
                filename='D.pth')


        #log for training process
        log_path = os.path.join(args.checkpoint, 'log.txt')
        # os.system('rm -rf {}'.format(log_path))
        with open(log_path, 'a+') as f:
            f.write(output_log + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--img_size', nargs='?', type=int, default=640, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=600, 
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[200, 400],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                        help='Batch Size per gpu')
    parser.add_argument('--lr_G', nargs='?', type=float, default=2.5e-4,
                        help='Learning Rate')
    parser.add_argument('--lr_D', nargs='?', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
    args = parser.parse_args()

    main(args)
