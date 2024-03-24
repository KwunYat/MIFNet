import sys
import os
import time
from optparse import OptionParser
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from eval import eval_net
from model.FasterNet import *
from model.MIFNetBlocks import *
from model.MIFNet import *
from utils import *
import tqdm

def train_net(net,
              epochs=200,
              batch_size=100,
              lr=0.005,
              val_percent=0.1,
              save_cp=True,
              gpu=True):
    dir_img = './datasets/38-Cloud_training/'
    dir_mask = './datasets/38-Cloud_training/train_gt/'
    dir_edge = './datasets/38-Cloud_training/train_edge/'
    dir_checkpoint = 'checkpoints/'


    """Returns a list of the ids in the directory"""
    id_image = get_ids(dir_mask)

    # 然后将数据分为训练集和验证集两份,返回字典
    iddataset = split_train_val(id_image, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=0.005,
                          momentum=0.9,
                          weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    criterion_1 = nn.BCELoss()
 
    for epoch in range(epochs):
        time1 = time.time()
        print('Starting epoch {}/{}. lr={}'.format(epoch + 1, epochs, lr))

        net.train()

        # reset the generators
        train = get_imgs_and_masks_and_edges(iddataset['train'], dir_img, dir_mask, dir_edge)
        val = get_imgs_and_masks_and_edges(iddataset['val'], dir_img, dir_mask, dir_edge)

        epoch_loss = 0

        for i, b in enumerate(tqdm.tqdm(batch(train, batch_size))):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b]).astype(np.float32)
            true_edges = np.array([i[1] for i in b]).astype(np.float32)

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            true_edges = torch.from_numpy(true_edges)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
                true_edges = true_edges.cuda()
                

            masks_pred = net(imgs)
            masks_pred0 = masks_pred[0]
            masks_pred1 = masks_pred[1]
            masks_pred2 = masks_pred[2]
            masks_pred3 = masks_pred[3]
            edge_pred1 = masks_pred[4]
            edge_pred2 = masks_pred[5]
            edge_pred3 = masks_pred[6]
            masks_probs_flat0 = masks_pred0.view(-1)
            masks_probs_flat1 = masks_pred1.view(-1)
            masks_probs_flat2 = masks_pred2.view(-1)
            masks_probs_flat3 = masks_pred3.view(-1)
            edges_probs_flat1 = edge_pred1.view(-1)
            edges_probs_flat2 = edge_pred2.view(-1)
            edges_probs_flat3 = edge_pred3.view(-1)
            true_masks_flat = true_masks.view(-1)
            true_edges_flat = true_edges.view(-1)


            loss0 = criterion_1(masks_probs_flat0, true_masks_flat)
            loss1 = criterion_1(masks_probs_flat1, true_masks_flat)
            loss2 = criterion_1(masks_probs_flat2, true_masks_flat)
            loss3 = criterion_1(masks_probs_flat3, true_masks_flat)
            loss4 = criterion_1(edges_probs_flat1, true_edges_flat)
            loss5 = criterion_1(edges_probs_flat2, true_edges_flat)
            loss6 = criterion_1(edges_probs_flat3, true_edges_flat)
            loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6

            epoch_loss += loss.item()

            #print('{0:d} / {1:d} --- loss: {2:.6f}'.format(i * batch_size, N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time2 = time.time() - time1
        print('Epoch finished ! Loss: {}, cost time: {}'.format(epoch_loss / i, time2))

        scheduler.step()
        lr = scheduler.get_lr()

        if 1:
            iou, oa, recall, precision, f1 = eval_net(net, val, gpu)
            print('mIoU: {0:.4f}, Overall Accuracy:{1:.4f}, Recall: {2:.4f}, Precision: {3:.4f}, F-measure: {4:.4f}'
                  .format(iou, oa, recall, precision, f1))

        if save_cp:
            torch.save(net,
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=64,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.005,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    net = MIFNet(weights = './model/pretrain/fasternet_s-epoch.299-val_acc1.81.2840.pth', cfg='./model/cfg/fasternet_s.yaml')
    net = nn.DataParallel(net.cuda(), device_ids=[0])


    if args.load:
        net = torch.load(args.load)
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu)
    except KeyboardInterrupt:
        #torch.save(net.state_dict(), 'INTERRUPTED.pth')
        torch.save(net, 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

