from __future__ import print_function, division
 
import argparse
import os,errno
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
import numpy as np
 
import torch.utils.model_zoo as model_zoo
 
from dataset import TSNDataSet, PathFolder
from models_cospcan import TSN, Discriminator_Weights_Adjust
from transforms import *
from opts import parser
 
from torchvision import datasets, models, transforms
 
import torch.nn.functional as F
 
import matplotlib
from matplotlib.offsetbox import AnchoredText
matplotlib.use('Agg')
 
import matplotlib.pyplot as plt
 
from operator import itemgetter
from PIL import Image, ImageDraw,ImageFont

import glob
# import pretrainedmodels

best_prec1 = 0

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
source_acc_point = []
domain_acc_point = []
target_acc_point = []

source_acc_point2 = []
domain_acc_point2 = []
target_acc_point2 = []

w2 = 0 
w3 = 0 
w4 = 0 
wm = 0 

w2_2 = 0 
w3_2 = 0 
w4_2 = 0 
wm_2 = 0 

threshold_count = 0 
epoch_acc_s = 0 
avg_epoch_acc_s = 0 
total_epoch_acc_s = 0 
pre_epoch_acc_s = 0 
prev_epoch_acc_s = 0
double_desc = 0 
train_num = 0

threshold_count2 = 0 
epoch_acc_s2 = 0 
avg_epoch_acc_s2 = 0 
total_epoch_acc_s2 = 0 
pre_epoch_acc_s2 = 0 
prev_epoch_acc_s2 = 0
double_desc2 = 0 
train_num2 = 0

def main():
    global args, best_prec1, w2, w3, w4, wm, threshold_count, epoch_acc_s, avg_epoch_acc_s, total_epoch_acc_s, pre_epoch_acc_s, double_desc, train_num
    global w2_2 , w3_2 , w4_2 , wm_2 , threshold_count2 , epoch_acc_s2 , avg_epoch_acc_s2 , total_epoch_acc_s2 , pre_epoch_acc_s2 , double_desc2 , train_num2
    global source_acc_point, domain_acc_point, target_acc_point, source_acc_point2, domain_acc_point2, target_acc_point
    # print("args.gpus",args.gpus)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus[0])

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'ucf101-10':
        num_class = 10
    elif args.dataset == 'hmdb51-10':
        num_class = 10
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    
    # model_pretrained = load_model_merged('bninception', args.pretrain_num_class) model_pretrained,

    init_w_main = Variable(torch.FloatTensor([float(args.main_w)]).cuda(0))
    init_w_l2 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda(0))
    init_w_l3 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda(0))
    init_w_l4 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda(0))

    init_w_main_2 = Variable(torch.FloatTensor([float(args.main_w)]).cuda(1))
    init_w_l2_2 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda(1))
    init_w_l3_2 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda(1))
    init_w_l4_2 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda(1))

    model = TSN(num_class, args.num_segments, args.modality, 
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn, 
                form_weight=args.form_w, last_weight=args.main_w, init_w_l2=init_w_l2, init_w_l3=init_w_l3, init_w_l4=init_w_l4, init_w_main=init_w_main)

    model2 = TSN(num_class, args.num_segments, args.modality2, 
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout2, partial_bn=not args.no_partialbn, 
                form_weight=args.form_w, last_weight=args.main_w, init_w_l2=init_w_l2_2, init_w_l3=init_w_l3_2, init_w_l4=init_w_l4_2, init_w_main=init_w_main_2)

    test_model = TSN(num_class, 1, args.modality,
                      base_model=args.arch,
                      consensus_type=args.crop_fusion_type,
                      dropout=args.test_dropout)

    test_model2 = TSN(num_class, 1, args.modality2,
                      base_model=args.arch,
                      consensus_type=args.crop_fusion_type,
                      dropout=args.test_dropout)
    print("model",model)
    print("form_w", args.form_w, "main_w", args.main_w)
    # disc_w_adjust = Discriminator_Weights_Adjust(args.form_w, args.main_w)
    # print("model",test_model)

    # print("test_model",test_model)
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    input_mean2 = model2.input_mean
    input_std2 = model2.input_std

    base_policies, source_bottle_policies, sub_bottle_policies, source_cls_policies, \
                                                   domain_cls_policies, disc_w_policies = model.get_optim_policies()


    base_policies2, source_bottle_policies2, sub_bottle_policies2, source_cls_policies2, \
                                                   domain_cls_policies2, disc_w_policies2 = model2.get_optim_policies()

    train_augmentation = model.get_augmentation()

    train_augmentation2 = model2.get_augmentation()

    # disc_w_policies = [model.l2_var.parameters(), model.l3_var.parameters()]

    # model = model.cuda()
    model = model.cuda(0)

    model2 = model2.cuda(1)

    test_model = test_model.cuda(0)

    test_model2 = test_model2.cuda(1)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
        normalize2 = GroupNormalize(input_mean2, input_std2)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    if args.modality2 == 'RGB':
        data_length2 = 1
    elif args.modality2 in ['Flow', 'RGBDiff']:
        data_length2 = 5
                                    
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(test_model.scale_size),
            GroupCenterCrop(test_model.input_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(test_model.input_size, test_model.scale_size)
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))


    transforms = {
                    'source': torchvision.transforms.Compose([
                           train_augmentation,
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           normalize,
                    ]),
                    'target': torchvision.transforms.Compose([
                           train_augmentation,
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           normalize,
                    ]),
                    'val': torchvision.transforms.Compose([
                           GroupScale(int(scale_size)),
                           GroupCenterCrop(crop_size),
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           normalize,
                    ]),
                    'test': torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           GroupNormalize(test_model.input_mean, test_model.input_std),
                    ]),
                    'source2': torchvision.transforms.Compose([
                           train_augmentation2,
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           normalize2,
                    ]),
                    'target2': torchvision.transforms.Compose([
                           train_augmentation2,
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           normalize2,
                    ]),
                    'val2': torchvision.transforms.Compose([
                           GroupScale(int(scale_size)),
                           GroupCenterCrop(crop_size),
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           normalize2,
                    ]),
                    'test2': torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=args.arch == 'BNInception'),
                           ToTorchFormatTensor(div=args.arch != 'BNInception'),
                           GroupNormalize(test_model2.input_mean, test_model2.input_std),
                   ])
                  }

    dsets = {}
    dsets['source'] = TSNDataSet("", args.train_list, num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                 transform=transforms['source'])

    dsets['target'] = TSNDataSet("", args.val_list, num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                 transform=transforms['target'])

    dsets['val'] = TSNDataSet("", args.val_list, num_segments=args.num_segments,
                             new_length=data_length,
                             modality=args.modality,
                             image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                             random_shift=False,
                             transform=transforms['val'])
    dsets['test'] = TSNDataSet("", args.val_list, num_segments=args.test_segments,
                               new_length=data_length,
                               modality=args.modality,
                               image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                               test_mode=True,
                               transform=transforms['test'])

    # -------------------- modality 2 ---------------------

    dsets['source2'] = TSNDataSet("", args.train_list2, num_segments=args.num_segments,
                                 new_length=data_length2,
                                 modality=args.modality2,
                                 image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                 transform=transforms['source2'])

    dsets['target2'] = TSNDataSet("", args.val_list, num_segments=args.num_segments,
                                 new_length=data_length2,
                                 modality=args.modality2,
                                 image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                 transform=transforms['target2'])

    dsets['val2'] = TSNDataSet("", args.val_list, num_segments=args.num_segments,
                             new_length=data_length2,
                             modality=args.modality2,
                             image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                             random_shift=False,
                             transform=transforms['val2'])
    dsets['test2'] = TSNDataSet("", args.val_list, num_segments=args.test_segments,
                               new_length=data_length2,
                               modality=args.modality2,
                               image_tmpl="img_{:05d}.jpg" if args.modality2 in ['RGB', 'RGBDiff'] else args.flow_prefix2+"{}_{:05d}.jpg",
                               test_mode=True,
                               transform=transforms['test2'])

    dsets['pseudo'] = []
    dsets['d_pseudo_all'] = []
    dsets['d_pseudo_all_feat'] = [] 

    dsets['pseudo2'] = []
    dsets['d_pseudo_all2'] = []
    dsets['d_pseudo_all_feat2'] = [] 
    # source_loader = torch.utils.data.DataLoader(dsets['source'],
    #     batch_size=int(args.batch_size/2), shuffle=True,
    #     num_workers=args.workers)
    
    # target_loader = torch.utils.data.DataLoader(dsets['target'],
    #     batch_size=int(args.batch_size/2), shuffle=True,
    #     num_workers=args.workers)

    # val_loader = torch.utils.data.DataLoader(dsets['val'],
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers)

    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=int(args.batch_size / 2),
                    shuffle=True) for x in ['source', 'target', 'source2', 'target2']}

    dset_loaders['val'] = torch.utils.data.DataLoader(dsets['val'], batch_size=1,
                    shuffle=False)

    dset_loaders['test'] = torch.utils.data.DataLoader(dsets['test'], batch_size=1,
                    shuffle=False)

    dset_loaders['val2'] = torch.utils.data.DataLoader(dsets['val2'], batch_size=1,
                    shuffle=False)

    dset_loaders['test2'] = torch.utils.data.DataLoader(dsets['test2'], batch_size=1,
                    shuffle=False)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
        domain_criterion = torch.nn.BCEWithLogitsLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in base_policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer_cls = torch.optim.SGD(base_policies+source_bottle_policies+source_cls_policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_dom = torch.optim.SGD(sub_bottle_policies+source_bottle_policies+domain_cls_policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_dom_feature = torch.optim.SGD(base_policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_dom_w = torch.optim.SGD(disc_w_policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_cls2 = torch.optim.SGD(base_policies2+source_bottle_policies2+source_cls_policies2,
                                args.lr2,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_dom2 = torch.optim.SGD(sub_bottle_policies2+source_bottle_policies2+domain_cls_policies2,
                                args.lr2,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_dom_feature2 = torch.optim.SGD(base_policies2,
                                args.lr2,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_dom_w2 = torch.optim.SGD(disc_w_policies2,
                                args.lr2,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # if args.evaluate:
    #     validate(dset_loaders['val'], model, criterion, domain_criterion, 0)
    #     validate(dset_loaders['val2'], model2, criterion, domain_criterion, 0)
    #     return

    threshold_count = 0
    epoch_acc_s = 0
    avg_epoch_acc_s = 0
    prev_epoch_acc_s = 0
    total_epoch_acc_s = 0
    pre_epoch_acc_s = 0
    double_desc = False

    threshold_count2 = 0
    epoch_acc_s2 = 0
    avg_epoch_acc_s2 = 0
    prev_epoch_acc_s2 = 0
    total_epoch_acc_s2 = 0
    pre_epoch_acc_s2 = 0
    double_desc2 = False

    # print("model", model)
    # print("model2", model2)

    for epoch in range(args.start_epoch, args.epochs):

        pre_epochs = int(args.pre_ratio * args.epochs)
        # args.pre_ratio
        p = epoch / (args.epochs * args.lr_ratio)
        # p = pull_back_to_one(p)
        decay = (1. + 10 * p)**(-0.75)

        adjust_learning_rate(optimizer_cls, args.lr, decay)
        adjust_learning_rate(optimizer_dom, args.lr, decay)
        adjust_learning_rate(optimizer_dom_feature, args.lr, decay)

        adjust_learning_rate(optimizer_cls2, args.lr2, decay)
        adjust_learning_rate(optimizer_dom2, args.lr2, decay)
        adjust_learning_rate(optimizer_dom_feature2, args.lr2, decay)

        disc_w_decay = args.wt * args.wp **p
        adjust_learning_rate(optimizer_dom_w, args.lr, disc_w_decay)

        adjust_learning_rate(optimizer_dom_w2, args.lr2, disc_w_decay)
                
        print()
        print("Epoch {} lr_decay: {} disc_w_decay: {}".format(epoch, decay, disc_w_decay))
        # optimizer_dom_w, epoch_lr_mult = dom_w_scheduler(optimizer_dom_w, decay, weight_mult)

        source_iter = iterator_reset('source', 'source', dsets, int(args.batch_size/2))
        target_iter = iterator_reset('target', 'target', dsets, int(args.batch_size/2))

        source_iter2 = iterator_reset('source2', 'source2', dsets, int(args.batch_size/2))
        target_iter2 = iterator_reset('target2', 'target2', dsets, int(args.batch_size/2))

        val_iter = iterator_reset('val','val', dsets, 1, shuffle_i=False)
        val_iter2 = iterator_reset('val2','val2', dsets, 1, shuffle_i=False)

        test_iter = iterator_reset('test','test', dsets, 1, shuffle_i=False)
        test_iter2 = iterator_reset('test2','test2', dsets, 1, shuffle_i=False)

        # ---------------- val ------------------------


        prec1, prec1_2, Pseudo_set, Pseudo_set2, Pseudo_set_co = validate(val_iter, val_iter2, model, model2, criterion, domain_criterion, (epoch + 1) * len(dset_loaders['val']), epoch)
        # prec1, prec1_2 = 0, 0
        # Pseudo_set, Pseudo_set2, Pseudo_set_co = [], [], []
        # --------------- test ------------------------

        for target_param, param in zip(test_model.parameters(), model.parameters()):
            target_param.data.copy_(param.data)

        for target_param2, param2 in zip(test_model2.parameters(), model2.parameters()):
            target_param2.data.copy_(param2.data)

        if epoch % 5 == 0:
            test_set, test_set2, test_set_co = test(test_iter, test_iter2, test_model, test_model2, criterion, domain_criterion, (epoch + 1) * len(dset_loaders['test']), epoch, num_class)

        # ----# ----# ----# ----# ----# ----# ----# ----

        train(dsets, transforms, dset_loaders, source_iter, target_iter, source_iter2, target_iter2, model, model2, Pseudo_set, Pseudo_set2, Pseudo_set_co, criterion, domain_criterion, optimizer_cls, optimizer_dom, optimizer_dom_w, optimizer_dom_feature, optimizer_cls2, optimizer_dom2, optimizer_dom_w2, optimizer_dom_feature2, epoch, pre_epochs, num_class)

        # save
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': prec1,
            }, mod=args.modality.lower())
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model2.state_dict(),
                'best_prec1': prec1_2,
            }, mod=args.modality2.lower())

        if epoch == 180:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': prec1,
            }, epoch=epoch, mod=args.modality.lower())
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model2.state_dict(),
                'best_prec1': prec1_2,
            }, epoch=epoch, mod=args.modality2.lower())


def cal_pseudo_label_set(model, pseudo_dataset):

    dset_loaders[pseudo_dataset] = torch.utils.data.DataLoader(dsets[pseudo_dataset], batch_size=1, shuffle=False)
            
    Pseudo_set_f = []
    pseudo_counter = 0

    for pseudo_inputs, pseudo_labels, pseudo_path, _ in dset_loaders[pseudo_dataset]:

        pseudo_inputs = Variable(pseudo_inputs.cuda())

        domain_labels_t = Variable(torch.FloatTensor([0.]*len(pseudo_inputs)).cuda())

        ini_weight = Variable(torch.FloatTensor([1.0]*len(pseudo_inputs)).cuda())

        class_t, domain_out_t, confid_rate = model('pseudo_discriminator', pseudo_inputs,[],[],domain_labels_t,ini_weight)

        # prediction confidence weight

        # domain variance

        dom_prob = F.sigmoid(domain_out_t.squeeze())

        top_prob, top_label = torch.topk(F.softmax(class_t.squeeze()), 1)

        # dom_confid = 1 - dom_prob.data[0]

        # s_tuple = (pseudo_path, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(pseudo_labels[0]))
        s_tuple = (pseudo_path, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(pseudo_labels[0]))
        Pseudo_set_f.append(s_tuple)
        # -------- sort domain variance score, reduce memory ------------
        fake_sample = int(int(top_label[0].cpu().data[0]) != int(pseudo_labels[0]))

        # total_pseudo_errors += fake_sample
        pseudo_counter += 1

    return Pseudo_set_f

# test_source
def cal_test_source_accuracy(model, test_source_dataset):

    test_source_corrects = 0
    dset_loaders[test_source_dataset] = torch.utils.data.DataLoader(dsets[test_source_dataset], batch_size=1, shuffle=False)
        
    for test_source_input, test_source_label in dset_loaders[test_source_dataset]:

        test_source_input, test_source_label = Variable(test_source_input.cuda()), Variable(test_source_label.cuda())
        test_source_outputs = model('test', test_source_input)

        # ------------ test classification statistics ------------
        _, test_source_preds = torch.max(test_source_outputs.data, 1)
        test_source_corrects += torch.sum(test_source_preds == test_source_label.data)

    # epoch_loss = epoch_loss / len(dsets['test'])
    #             epoch_acc = epoch_corrects / len(dsets['test'])
    #             epoch_acc_t = epoch_acc

    acc_test_source = test_source_corrects / len(dsets[test_source_dataset])

    return acc_test_source

def iterator_reset(iterator_name_i, dset_name_i, dsets, batchsize_i, shuffle_i=True, drop_last_i=True):
    dset_loaders = {}
    if batchsize_i > 0:
        if args.useRatio:
            dset_loaders[iterator_name_i] = torch.utils.data.DataLoader(dsets[dset_name_i],
                                                batch_size=1, shuffle=shuffle_i)
        else:
            dset_loaders[iterator_name_i] = torch.utils.data.DataLoader(dsets[dset_name_i],
                                                batch_size=batchsize_i, shuffle=shuffle_i, drop_last=drop_last_i, pin_memory=True)
        iterator = iter(dset_loaders[iterator_name_i])
    else:
        iterator = iter([])
    
    return iterator

def iterator_update(iterator_i, dset_name_i, dsets, batchsize_i, pointer_i, total_pointer_i,  i_type="pseu"):
    if args.useRatio:
        if pointer_i + batchsize_i > len(iterator_i):
            iterator_i = iterator_reset(iterator_i, dset_name_i, dsets, batchsize_i)
            pointer_i = 0

        iterator_batch_inputs = torch.FloatTensor([])
        iterator_batch_labels = torch.LongTensor([])
        iterator_batch_weights = torch.DoubleTensor([])
        iterator_batch_dom_confid = torch.DoubleTensor([])

        for i in range(batchsize_i):
            if i_type == "pseu":
                i_inputs, i_labels, _, i_dom_conf, i_weights, _, _ = iterator_i.next()
                iterator_batch_weights = torch.cat((iterator_batch_weights,i_weights), 0)
                iterator_batch_dom_confid = torch.cat((iterator_batch_dom_confid,(1-i_dom_conf)), 0)
            else:
                i_inputs, i_labels, _, _ = iterator_i.next()

            iterator_batch_inputs = torch.cat((iterator_batch_inputs,i_inputs), 0)
            iterator_batch_labels = torch.cat((iterator_batch_labels,i_labels), 0)

            pointer_i += 1
            total_pointer_i += 1
    else:
        if pointer_i +1 > len(iterator_i):
            iterator_i = iterator_reset(iterator_i, dset_name_i, dsets,  batchsize_i)
            pointer_i = 0

        if i_type == "pseu":
            iterator_batch_inputs, iterator_batch_labels, _, i_dom_conf, iterator_batch_weights, _, _  = iterator_i.next()
            iterator_batch_dom_confid = 2*(1 - i_dom_conf)
        else:
            iterator_batch_inputs, iterator_batch_labels, _, _ = iterator_i.next()

        pointer_i += 1
        total_pointer_i += 1

    if i_type == "pseu":
        return iterator_i, iterator_batch_inputs, iterator_batch_labels, pointer_i, total_pointer_i, iterator_batch_weights, iterator_batch_dom_confid
    else:
        return iterator_i, iterator_batch_inputs, iterator_batch_labels, pointer_i, total_pointer_i

def train(dsets, transforms, dset_loaders, train_source_iter, train_target_iter, train_source_iter2, train_target_iter2, model, model2,
            Pseudo_set, Pseudo_set2, Pseudo_set_co, criterion, domain_criterion, optimizer, dom_optimizer, dom_w_optimizer, dom_feat_optimizer, optimizer2, dom_optimizer2, dom_w_optimizer2, dom_feat_optimizer2, epoch, pre_epochs, num_class):
    global w2, w3, w4, wm, threshold_count, epoch_acc_s, avg_epoch_acc_s, prev_epoch_acc_s, total_epoch_acc_s, pre_epoch_acc_s, double_desc, train_num
    global w2_2, w3_2, w4_2, wm_2, threshold_count2, epoch_acc_s2, avg_epoch_acc_s2, prev_epoch_acc_s2, total_epoch_acc_s2, pre_epoch_acc_s2, double_desc2, train_num2
    global source_acc_point, domain_acc_point, target_acc_point, source_acc_point2, domain_acc_point2, target_acc_point

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    domain_meter = AverageMeter()

    losses_2 = AverageMeter()
    top1_2 = AverageMeter()
    domain_meter_2 = AverageMeter()


    if args.no_partialbn:
        model.partialBN(False)
        model2.partialBN(False)
        # module.
    else:
        model.partialBN(True)
        model2.partialBN(True)

    # switch to train mode
    model.train()
    model2.train()

    source_iter = train_source_iter
    d_target_iter = train_target_iter

    source_iter2 = train_source_iter2
    d_target_iter2 = train_target_iter2

    # ---- classifier -----
    source_pointer = 0
    pseudo_pointer = 0

    total_source_pointer = 0
    total_pseudo_pointer = 0

    source_pointer2 = 0
    pseudo_pointer2 = 0

    total_source_pointer2 = 0
    total_pseudo_pointer2 = 0

    # ---- domain discriminator -----

    # ------- source part --------
    d_source_pointer = 0
    d_source_feat_pointer = 0

    d_pseudo_source_pointer = 0
    d_pseudo_source_feat_pointer = 0

    total_d_source_pointer = 0
    total_d_source_feat_pointer = 0

    total_d_pseudo_source_pointer = 0
    total_d_pseudo_source_feat_pointer = 0

    # ------- target part --------
    d_target_pointer = 0
    d_target_feat_pointer = 0

    d_pseudo_pointer = 0
    d_pseudo_feat_pointer = 0
    
    d_pseudo_all_pointer = 0
    d_pseudo_all_feat_pointer = 0

    total_d_target_pointer = 0
    total_d_target_feat_pointer = 0

    total_d_pseudo_pointer = 0
    total_d_pseudo_feat_pointer = 0

    total_d_pseudo_all_pointer = 0
    total_d_pseudo_all_feat_pointer = 0

    # ----------------------------

    domain_epoch_corrects = 0
    total_epoch_loss = 0

    # ----------------------------

    batch_count = 0
    class_count = 0
    domain_counts = 0

    confid_threshold = 0

    # ---------------------------------------
    # ------- modality 2 source part --------
    # ---------------------------------------

    d_source_pointer2 = 0
    d_source_feat_pointer2 = 0

    d_pseudo_source_pointer2 = 0
    d_pseudo_source_feat_pointer2 = 0

    total_d_source_pointer2 = 0
    total_d_source_feat_pointer2 = 0

    total_d_pseudo_source_pointer2 = 0
    total_d_pseudo_source_feat_pointer2 = 0

    # ------- target part --------
    d_target_pointer2 = 0
    d_target_feat_pointer2 = 0

    d_pseudo_pointer2 = 0
    d_pseudo_feat_pointer2 = 0
    
    d_pseudo_all_pointer2 = 0
    d_pseudo_all_feat_pointer2 = 0

    total_d_target_pointer2 = 0
    total_d_target_feat_pointer2 = 0

    total_d_pseudo_pointer2 = 0
    total_d_pseudo_feat_pointer2 = 0

    total_d_pseudo_all_pointer2 = 0
    total_d_pseudo_all_feat_pointer2 = 0

    # ----------------------------

    domain_epoch_corrects2 = 0
    total_epoch_loss2 = 0

    # ----------------------------

    batch_count2 = 0
    class_count2 = 0
    domain_counts2 = 0

    confid_threshold2 = 0

    # -----------------------------------------
    #  ---------- test preprocessing ----------




    # ------- iterative pseudo sample filter --------    
    if epoch >= pre_epochs and epoch % args.skip == 0:

        pseudo_time = time.time()
        # ----------------- sort pseudo datasets -------------------

        pseudo_num = len(Pseudo_set)
        Sorted_confid_set = sorted(Pseudo_set, key=lambda tup: tup[2], reverse=True)

        print("sort pseudo time:",time.time() - pseudo_time)

        # ----------------- sort pseudo2 datasets -------------------

        pseudo_num2 = len(Pseudo_set2)
        Sorted_confid_set2 = sorted(Pseudo_set2, key=lambda tup: tup[2], reverse=True)

        print("sort pseudo2 time:",time.time() - pseudo_time)

        # ----------------- sort pseudo co datasets -------------------

        pseudo_num_co = len(Pseudo_set_co)
        Sorted_confid_set_co = sorted(Pseudo_set_co, key=lambda tup: tup[2], reverse=True)

        print("sort pseudo_co time:",time.time() - pseudo_time)

        # ----------------- calculate pseudo threshold -------------------
        threshold_count += 1
        threshold_count2 += 1

        if args.ReTestSource:
            current_test_source_acc = cal_test_source_accuracy(model, 'test_source')
            max_threshold = args.totalPseudoChange
        else:
            current_test_source_acc = epoch_acc_s
            current_test_source_acc2 = epoch_acc_s2
            if args.usePrevAcc:
                avg_test_source_acc = prev_epoch_acc_s
                avg_test_source_acc2 = prev_epoch_acc_s2
            else:
                avg_test_source_acc = avg_epoch_acc_s
                avg_test_source_acc2 = avg_epoch_acc_s2
            max_threshold = args.epochs
            max_threshold2 = args.epochs2

        if current_test_source_acc < avg_test_source_acc:
            if not double_desc:
                double_desc = True

                if args.usingTriDec:
                    threshold_count -= 2
                elif args.usingDoubleDecrease:
                    threshold_count -= 1
            else:
                if args.usingTriDec:
                    threshold_count -= 3
                else:
                    threshold_count -= 2
        else:
            double_desc = False

        if current_test_source_acc2 < avg_test_source_acc2:
            if not double_desc2:
                double_desc2 = True

                if args.usingTriDec2:
                    threshold_count2 -= 2
                elif args.usingDoubleDecrease2:
                    threshold_count2 -= 1
            else:
                if args.usingTriDec2:
                    threshold_count2 -= 3
                else:
                    threshold_count2 -= 2
        else:
            double_desc2 = False

        print("current_test_source_acc", current_test_source_acc, "avg_test_source_acc", avg_test_source_acc, "prev_epoch_acc_s",prev_epoch_acc_s, "double_desc", double_desc)

        print("current_test_source_acc2", current_test_source_acc2, "avg_test_source_acc2", avg_test_source_acc2, "prev_epoch_acc_s2",prev_epoch_acc_s2, "double_desc2", double_desc2)        

        # -----------------------------------------------------------

        # sourceTestIter
        # totalPseudoChange

        pseu_n = threshold_count / max_threshold
        pseu_n = pull_back_to_one(pseu_n) * args.pseudo_ratio

        pseu_n = pull_back_to_n(pseu_n, args.max_pseudo)
        pseu_n = args.defaultPseudoRatio + pseu_n

        if pseu_n < 0:
            pseu_n = 0
        if pseu_n > 1:
            pseu_n = 1

        print("threshold_count: {} pseudo_ratio:{:.4f}".format(threshold_count, pseu_n))

        t_p = pseu_n

        # ----------------------------------------------------
        pseu_n2 = threshold_count2 / max_threshold2
        pseu_n2 = pull_back_to_one(pseu_n2) * args.pseudo_ratio2

        pseu_n2 = pull_back_to_n(pseu_n2, args.max_pseudo2)
        pseu_n2 = args.defaultPseudoRatio2 + pseu_n2

        if pseu_n2 < 0:
            pseu_n2 = 0
        if pseu_n2 > 1:
            pseu_n2 = 1
            
        print("threshold2_count: {} pseudo2_ratio:{:.4f}".format(threshold_count2, pseu_n2))

        t_p2 = pseu_n2

        # ----------------- get pseudo_C set -------------------

        confid_pseudo_num = int(t_p*pseudo_num_co)
        Select_confid_set = Sorted_confid_set_co[:confid_pseudo_num]

        if len(Select_confid_set) > 0:
            confid_threshold = Select_confid_set[-1][2]

        Select_T1_set = Select_confid_set


        # ----------------- get pseudo_C set 2 -------------------
        
        confid_pseudo_num2 = int(t_p2*pseudo_num_co)
        Select_confid_set2 = Sorted_confid_set_co[:confid_pseudo_num2]

        if len(Select_confid_set2) > 0:
            confid_threshold2 = Select_confid_set2[-1][2]

        Select_T1_set2 = Select_confid_set2


        # ----------------- get pseudo_D set (T1 + T2)--------------
        Sorted_dom_set = sorted(Pseudo_set, key=lambda tup: abs(tup[3] - 0.5), reverse=True)
        
        t_d = pseu_n

        domain_threshold_num = int(t_d*pseudo_num)
        Select_T2_set = Sorted_dom_set[:domain_threshold_num] # near 0 and 1

        Selected_T2_Minus_T1_set = [(i[0], i[1], i[2], i[3], i[4], i[5], i[6]) \
                               for i in Select_T2_set if i[2] < confid_threshold] # T2 - T1

        Select_T1T2_set = Select_T1_set + Selected_T2_Minus_T1_set # T1 Union T2

        # ----------------- get pseudo_D set 2 (T1_2 + T2_2)--------------
        Sorted_dom_set2 = sorted(Pseudo_set2, key=lambda tup: abs(tup[3] - 0.5), reverse=True)
        
        t_d2 = pseu_n2

        domain_threshold_num2 = int(t_d2*pseudo_num2)
        Select_T2_set2 = Sorted_dom_set2[:domain_threshold_num2] # near 0 and 1

        Selected_T2_Minus_T1_set2 = [(i[0], i[1], i[2], i[3], i[4], i[5], i[6]) \
                               for i in Select_T2_set2 if i[2] < confid_threshold2] # T2 - T1

        Select_T1T2_set2 = Select_T1_set2 + Selected_T2_Minus_T1_set2 # T1 Union T2


        
        # ----------------- get pseudo datasets Cross train-------------------


        if args.modality == 'RGB':
            data_length = 1
        elif args.modality in ['Flow', 'RGBDiff']:
            data_length = 5

        if len(Select_T1_set) > 0:
            dsets['pseudo'] = PathFolder(Select_T1_set, num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
        if len(Select_T2_set) > 0:
            dsets['d_pseudo_target'] = PathFolder(Select_T2_set, num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                 transform=transforms['target'])

        if len(Select_T2_set) > 0:
            dsets['d_pseudo_target_feat'] = PathFolder(Select_T2_set, num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
            
        # if len(Select_T3_set) > 0:
        #     dsets['d_pseudo_target_feat'] = PathFolder(Select_T3_set, data_transforms['target'])

        if len(Select_T1T2_set) > 0:
            if args.useT1Only:
                dsets['d_pseudo_all'] = PathFolder(Select_T1_set, num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
                dsets['d_pseudo_all_feat'] = PathFolder(Select_T1_set, num_segments=args.num_segments,
                                     new_length=data_length,
                                     modality=args.modality,
                                     image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                     transform=transforms['target'])
            else:
                dsets['d_pseudo_all'] = PathFolder(Select_T1T2_set, num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
                dsets['d_pseudo_all_feat'] = PathFolder(Select_T1T2_set, num_segments=args.num_segments,
                                 new_length=data_length,
                                 modality=args.modality,
                                 image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
        
        # -------------------------------------------------------

        if args.modality2 == 'RGB':
            data_length2 = 1
        elif args.modality2 in ['Flow', 'RGBDiff']:
            data_length2 = 5

        if len(Select_T1_set2) > 0:
            dsets['pseudo2'] = PathFolder(Select_T1_set2, num_segments=args.num_segments,
                                 new_length=data_length2,
                                 modality=args.modality2,
                                 image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
        if len(Select_T2_set2) > 0:
            dsets['d_pseudo_target2'] = PathFolder(Select_T2_set2, num_segments=args.num_segments,
                                 new_length=data_length2,
                                 modality=args.modality2,
                                 image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                 transform=transforms['target'])

        if len(Select_T2_set2) > 0:
            dsets['d_pseudo_target_feat2'] = PathFolder(Select_T2_set2, num_segments=args.num_segments,
                                 new_length=data_length2,
                                 modality=args.modality2,
                                 image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
            
        # if len(Select_T3_set) > 0:
        #     dsets['d_pseudo_target_feat'] = PathFolder(Select_T3_set, data_transforms['target'])

        if len(Select_T1T2_set2) > 0:
            if args.useT1Only:
                dsets['d_pseudo_all2'] = PathFolder(Select_T1_set2, num_segments=args.num_segments,
                                 new_length=data_length2,
                                 modality=args.modality2,
                                 image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
                dsets['d_pseudo_all_feat2'] = PathFolder(Select_T1_set2, num_segments=args.num_segments,
                                     new_length=data_length2,
                                     modality=args.modality2,
                                     image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                     transform=transforms['target'])
            else:
                dsets['d_pseudo_all2'] = PathFolder(Select_T1T2_set2, num_segments=args.num_segments,
                                 new_length=data_length2,
                                 modality=args.modality2,
                                 image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                 transform=transforms['target'])
                dsets['d_pseudo_all_feat2'] = PathFolder(Select_T1T2_set2, num_segments=args.num_segments,
                                 new_length=data_length2,
                                 modality=args.modality2,
                                 image_tmpl="img_{:05d}.jpg" if args.modality2 in ["RGB", "RGBDiff"] else args.flow_prefix2+"{}_{:05d}.jpg",
                                 transform=transforms['target'])                
        # ----------------- reload pseudo set ---------------------------

        # confid_threshold_point.append(float("%.4f" % confid_threshold))

    # free memory
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # ------- Source + Pseudo + Pseudo Target Dataset Preparation ---------
    # ---------------------------------------------------------------------

    # -------- reset source and pseudo batch ratio -------

    prep_starttime = time.time()

    source_size = len(dsets['source'])
    pseudo_size = len(dsets['pseudo'])
    d_target_size = len(dsets['target'])
    d_pseudo_all_size = len(dsets['d_pseudo_all'])

    total_errors = 0
    for ss in dsets['d_pseudo_all']:
        fake_sample = int(ss[1] != ss[-2])
        total_errors += fake_sample 

    print("Select error/total selected = {}/{}".format(total_errors,d_pseudo_all_size)) 

    # print("pseudo_all", [i[0] for i in dsets['d_pseudo_all']])

    if pseudo_size == 0:
        source_batchsize = int(args.batch_size / 2) 
        pseudo_batchsize = 0
        d_source_batchsize = int(args.batch_size / 2)
        d_target_batchsize = int(args.batch_size / 2)
        d_pseudo_source_batchsize = 0
    else:
        # source_batchsize = int(round(float(args.batch_size / 2) * source_size / float(source_size + pseudo_size)))
        source_batchsize = int(int(args.batch_size / 2) * source_size / (source_size + pseudo_size))

        if source_batchsize < int(int(args.batch_size / 2) / 2):
            source_batchsize = int(int(args.batch_size / 2) / 2)
        if source_batchsize == int(args.batch_size / 2):
            source_batchsize -= 1

        pseudo_batchsize = int(args.batch_size / 2) - source_batchsize
    
        d_source_batchsize = source_batchsize
        d_pseudo_source_batchsize = pseudo_batchsize

        d_pseudo_all_batchsize = 0
        if d_pseudo_all_size > 0:
            d_pseudo_all_batchsize = int(round(int(args.batch_size / 2) * d_pseudo_all_size / d_target_size))

            if d_pseudo_all_batchsize == 0:
                d_pseudo_all_batchsize = 1

        d_target_batchsize = int(args.batch_size / 2) - d_pseudo_all_batchsize

        pseudo_iter = iterator_reset('pseudo', 'pseudo', dsets, pseudo_batchsize)

        d_pseudo_source_iter = iterator_reset('d_pseudo_source', 'source', dsets, d_pseudo_source_batchsize)
        d_pseudo_source_feat_iter = iterator_reset('d_pseudo_source_feat', 'source', dsets, d_pseudo_source_batchsize)

        if d_pseudo_all_size > 0:

            d_pseudo_all_iter = iterator_reset('d_pseudo_all', 'd_pseudo_all', dsets, d_pseudo_all_batchsize)
            d_pseudo_all_feat_iter = iterator_reset('d_pseudo_all_feat', 'd_pseudo_all_feat', dsets, d_pseudo_all_batchsize)


    source_iter = iterator_reset('source','source', dsets, source_batchsize)

    d_source_iter = iterator_reset('d_source','source', dsets, d_source_batchsize)
    d_source_feat_iter = iterator_reset('d_source_feat','source', dsets, d_source_batchsize)

    d_target_iter = iterator_reset('d_target','target', dsets, d_target_batchsize)
    d_target_feat_iter = iterator_reset('d_target_feat','target', dsets, d_target_batchsize)

    # ------------------------------------------------------------------
    # -------- reset source and pseudo batch ratio for modality 2-------

    source_size2 = len(dsets['source2'])
    pseudo_size2 = len(dsets['pseudo2'])
    d_target_size2 = len(dsets['target2'])
    d_pseudo_all_size2 = len(dsets['d_pseudo_all2'])

    total_errors2 = 0
    for ss in dsets['d_pseudo_all2']:
        fake_sample = int(ss[1] != ss[-2])
        total_errors2 += fake_sample 

    print("Select2 error/total selected2 = {}/{}".format(total_errors2,d_pseudo_all_size2)) 

    # print("pseudo_all", [i[0] for i in dsets['d_pseudo_all']])

    if pseudo_size2 == 0:
        source_batchsize2 = int(args.batch_size / 2) 
        pseudo_batchsize2 = 0
        d_source_batchsize2 = int(args.batch_size / 2)
        d_target_batchsize2 = int(args.batch_size / 2)
        d_pseudo_source_batchsize2 = 0
    else:
        # source_batchsize = int(round(float(args.batch_size / 2) * source_size / float(source_size + pseudo_size)))
        source_batchsize2 = int(int(args.batch_size / 2) * source_size2 / (source_size2 + pseudo_size2))

        if source_batchsize2 < int(int(args.batch_size / 2) / 2):
            source_batchsize2 = int(int(args.batch_size / 2) / 2)
        if source_batchsize2 == int(args.batch_size / 2):
            source_batchsize2 -= 1

        pseudo_batchsize2 = int(args.batch_size / 2) - source_batchsize2
    
        d_source_batchsize2 = source_batchsize2
        d_pseudo_source_batchsize2 = pseudo_batchsize2

        d_pseudo_all_batchsize2 = 0
        if d_pseudo_all_size2 > 0:
            d_pseudo_all_batchsize2 = int(round(int(args.batch_size / 2) * d_pseudo_all_size2 / d_target_size2))

            if d_pseudo_all_batchsize2 == 0:
                d_pseudo_all_batchsize2 = 1

        d_target_batchsize2 = int(args.batch_size / 2) - d_pseudo_all_batchsize2

        pseudo_iter2 = iterator_reset('pseudo2', 'pseudo2', dsets, pseudo_batchsize2)

        d_pseudo_source_iter2 = iterator_reset('d_pseudo_source2', 'source2', dsets, d_pseudo_source_batchsize2)
        d_pseudo_source_feat_iter2 = iterator_reset('d_pseudo_source_feat2', 'source2', dsets, d_pseudo_source_batchsize2)

        if d_pseudo_all_size2 > 0:

            d_pseudo_all_iter2 = iterator_reset('d_pseudo_all2', 'd_pseudo_all2', dsets, d_pseudo_all_batchsize2)
            d_pseudo_all_feat_iter2 = iterator_reset('d_pseudo_all_feat2', 'd_pseudo_all_feat2', dsets, d_pseudo_all_batchsize2)


    source_iter2 = iterator_reset('source2','source2', dsets, source_batchsize2)

    d_source_iter2 = iterator_reset('d_source2','source2', dsets, d_source_batchsize2)
    d_source_feat_iter2 = iterator_reset('d_source_feat2','source2', dsets, d_source_batchsize2)

    d_target_iter2 = iterator_reset('d_target2','target2', dsets, d_target_batchsize2)
    d_target_feat_iter2 = iterator_reset('d_target_feat2','target2', dsets, d_target_batchsize2)


    prep_time = time.time() - prep_starttime

    print("data batch prep_time:", prep_time)

    # pseudo_size = len(dsets['pseudo'])

    # if pseudo_size == 0:
    #     source_batchsize = int(args.batch_size / 2) 
    #     pseudo_batchsize = 0
    #     d_source_batchsize = int(args.batch_size / 2)
    #     d_target_batchsize = int(args.batch_size / 2)

    i = 0
    confid_threshold = 0
    confid_threshold2 = 0
    prev_epoch_acc_s = epoch_acc_s
    prev_epoch_acc_s2 = epoch_acc_s2

    model.train(True)
    model2.train(True)

    end = time.time()

    while source_pointer < len(source_iter):

    # for i, (input, target) in enumerate(source_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        # --------------------- ------------------------- -----------------------
        # --------------------- classification part batch -----------------------
        # --------------------- ------------------------- -----------------------
        # if epoch < pre_epochs:
        #     source_batchsize = int(args.batch_size / 2)
        #     pseudo_batchsize = 0

        # else:
        #     source_size = len(dsets['source'])
        #     pseudo_size = len(dsets['pseudo'])
        #     pseudo_size2 = len(dsets['pseudo2'])

        #     source_batchsize = int(int(args.batch_size / 2) * source_size
        #                                     / (source_size + pseudo_size))

        #     source_batchsize2 = int(int(args.batch_size / 2) * source_size
        #                                     / (source_size + pseudo_size2))

        #     if pseudo_size > 0:
        #         if source_batchsize < int(int(args.batch_size / 2) / 2):
        #             source_batchsize = int(int(args.batch_size / 2) / 2)

        #         if source_batchsize == int(args.batch_size / 2) and epoch >= pre_epochs:
        #             source_batchsize -= 1

        #     pseudo_batchsize = int(args.batch_size / 2) - source_batchsize


        source_iter, inputs, labels, source_pointer, total_source_pointer \
                                    = iterator_update(source_iter, 'source', dsets, source_batchsize, 
                                                      source_pointer, total_source_pointer, "ori")

        source_iter2, inputs2, labels2, source_pointer2, total_source_pointer2 \
                                    = iterator_update(source_iter2, 'source2', dsets, source_batchsize2, 
                                                      source_pointer2, total_source_pointer2, "ori")

        if pseudo_batchsize > 0:
        
            pseudo_iter, pseudo_inputs, pseudo_labels, pseudo_pointer, total_pseudo_pointer, pseudo_weights, pseudo_dom_conf \
                            = iterator_update(pseudo_iter, 'pseudo', dsets,
                                               pseudo_batchsize, pseudo_pointer, 
                                               total_pseudo_pointer, "pseu")

        if pseudo_batchsize2 > 0:

            pseudo_iter2, pseudo_inputs2, pseudo_labels2, pseudo_pointer2, total_pseudo_pointer2, pseudo_weights2, pseudo_dom_conf2 \
                            = iterator_update(pseudo_iter2, 'pseudo2', dsets,
                                               pseudo_batchsize2, pseudo_pointer2, 
                                               total_pseudo_pointer2, "pseu")
        # --------------------- ------------------------------- -----------------------
        # --------------------- domain discriminator part batch -----------------------
        # --------------------- ------------------------------- -----------------------
        # if epoch < pre_epochs:
        #     d_pseudo_source_batchsize = 0
        #     d_source_batchsize = int(args.batch_size / 2)
        #     d_target_batchsize = int(args.batch_size / 2)

        # else:
        #     d_source_size = len(dsets['source'])
        #     d_pseudo_size = len(dsets['pseudo'])
        #     d_target_size = len(dsets['target'])
        #     d_pseudo_all_size = len(dsets['d_pseudo_all'])

        #     d_source_batchsize = source_batchsize
        #     d_pseudo_source_batchsize = pseudo_batchsize

        #     d_pseudo_all_batchsize = 0
        #     if d_pseudo_all_size > 0:
        #         d_pseudo_all_batchsize = int(round(int(args.batch_size / 2) * d_pseudo_all_size / d_target_size))

        #         if d_pseudo_all_batchsize == 0:
        #             d_pseudo_all_batchsize = 1

        #     d_target_batchsize = int(args.batch_size / 2) - d_pseudo_all_batchsize

        # ----------------- get domain discriminator input --------------------------

        d_source_iter, d_inputs, _, d_source_pointer, total_d_source_pointer \
                                    = iterator_update(d_source_iter, 'source', dsets, d_source_batchsize, 
                                                      d_source_pointer, total_d_source_pointer, "ori")
                                    # = source_iter, inputs, labels, source_pointer, total_source_pointer
        # d_source_feat_iter, d_feat_inputs, _, d_source_feat_pointer, total_d_source_feat_pointer \
        #                             = d_source_iter, d_inputs, _, d_source_pointer, total_d_source_pointer
                                    # = iterator_update(d_source_feat_iter, 'source', dsets, d_source_batchsize, 
                                    #                   d_source_feat_pointer, total_d_source_feat_pointer, "ori")

        if d_target_batchsize > 0:
            d_target_iter, d_target_inputs, _, d_target_pointer, total_d_target_pointer \
                            = iterator_update(d_target_iter, 'target', dsets, d_target_batchsize, 
                                              d_target_pointer, total_d_target_pointer, "ori")

            # d_target_feat_iter, d_target_feat_inputs, _, d_target_feat_pointer, total_d_target_feat_pointer \
            #                 = iterator_update(d_target_feat_iter, 'target', dsets, d_target_batchsize, 
            #                                   d_target_feat_pointer, total_d_target_feat_pointer, "ori")

        # ----------------- get domain discriminator input 2 --------------------------

        d_source_iter2, d_inputs2, _, d_source_pointer2, total_d_source_pointer2 \
                                    = iterator_update(d_source_iter2, 'source2', dsets, d_source_batchsize2, 
                                                      d_source_pointer2, total_d_source_pointer2, "ori")
                                    # = source_iter2, inputs2, labels2, source_pointer2, total_source_pointer2

        # d_source_feat_iter, d_feat_inputs, _, d_source_feat_pointer, total_d_source_feat_pointer \
        #                             = iterator_update(d_source_feat_iter, 'source', dsets, d_source_batchsize, 
        #                                               d_source_feat_pointer, total_d_source_feat_pointer, "ori")

        if d_target_batchsize2 > 0:
            d_target_iter2, d_target_inputs2, _, d_target_pointer2, total_d_target_pointer2 \
                            = iterator_update(d_target_iter2, 'target2', dsets, d_target_batchsize2, 
                                              d_target_pointer2, total_d_target_pointer2, "ori")

            # d_target_feat_iter, d_target_feat_inputs, _, d_target_feat_pointer, total_d_target_feat_pointer \
            #                 = iterator_update(d_target_feat_iter, 'target', dsets, d_target_batchsize, 
            #                                   d_target_feat_pointer, total_d_target_feat_pointer, "ori")


        # ----------------- get domain pseudo input --------------------------
        if d_pseudo_source_batchsize > 0:

            d_pseudo_source_iter, d_pseudo_source_inputs, _, d_pseudo_source_pointer, total_d_pseudo_source_pointer\
                        = iterator_update(d_pseudo_source_iter, 'd_pseudo_source', dsets,
                                           d_pseudo_source_batchsize, d_pseudo_source_pointer, 
                                           total_d_pseudo_source_pointer, "ori")
            
            # d_pseudo_source_feat_iter, d_pseudo_source_feat_inputs, _, d_pseudo_source_feat_pointer, total_d_pseudo_source_feat_pointer \
            #             = iterator_update(d_pseudo_source_feat_iter, 'd_pseudo_source', dsets,
            #                                d_pseudo_source_batchsize, d_pseudo_source_feat_pointer, 
            #                                total_d_pseudo_source_feat_pointer, "ori")
            
            # ------------------------ T1+T2 --------------------
            d_pseudo_all_iter, d_pseudo_all_inputs, _, d_pseudo_all_pointer, total_d_pseudo_all_pointer, _, d_pseudo_all_dom_conf \
                    = iterator_update(d_pseudo_all_iter, 'd_pseudo_all', dsets,
                                       d_pseudo_all_batchsize, d_pseudo_all_pointer, 
                                       total_d_pseudo_all_pointer, "pseu")

            # ------------------------ T1+T3 --------------------
            # d_pseudo_all_feat_iter, d_pseudo_all_feat_inputs, _, d_pseudo_all_feat_pointer, total_d_pseudo_all_feat_pointer, _, d_pseudo_all_feat_dom_conf \
            #         = iterator_update(d_pseudo_all_feat_iter, 'd_pseudo_all_feat', dsets, 
            #                            d_pseudo_all_batchsize, d_pseudo_all_feat_pointer, 
            #                            total_d_pseudo_all_feat_pointer, "pseu")
            
        if d_pseudo_source_batchsize2 > 0:
            d_pseudo_source_iter2, d_pseudo_source_inputs2, _, d_pseudo_source_pointer2, total_d_pseudo_source_pointer2\
                        = iterator_update(d_pseudo_source_iter2, 'd_pseudo_source2', dsets,
                                           d_pseudo_source_batchsize2, d_pseudo_source_pointer2, 
                                           total_d_pseudo_source_pointer2, "ori")
            
            # ------------------------ T1+T2 --------------------
            d_pseudo_all_iter2, d_pseudo_all_inputs2, _, d_pseudo_all_pointer2, total_d_pseudo_all_pointer2, _, d_pseudo_all_dom_conf2 \
                    = iterator_update(d_pseudo_all_iter2, 'd_pseudo_all2', dsets,
                                       d_pseudo_all_batchsize2, d_pseudo_all_pointer2, 
                                       total_d_pseudo_all_pointer2, "pseu")

        # --------------------- ------------------------------- -----------------------
        # ----------------------------- fit model ------------- -----------------------
        # --------------------- ------------------------------- -----------------------

        if epoch < pre_epochs or pseudo_batchsize <= 0:
            # ----------- classifier inputs----------
            fuse_inputs = inputs
            fuse_labels = labels

            fuse_inputs2 = inputs2
            fuse_labels2 = labels2

            # ----------- domain inputs----------
            domain_inputs = torch.cat((d_inputs, d_target_inputs),0)
            domain_inputs2 = torch.cat((d_inputs2, d_target_inputs2),0)

            domain_labels = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                             +[0.]*int(args.batch_size / 2))
            domain_labels2 = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                             +[0.]*int(args.batch_size / 2))

            dom_feat_weight = torch.FloatTensor([1.]*int(args.batch_size))
            dom_feat_weight2 = torch.FloatTensor([1.]*int(args.batch_size))

        else:
            # ----------- classifier inputs----------
            fuse_inputs = torch.cat((inputs, pseudo_inputs),0)
            fuse_labels = torch.cat((labels, pseudo_labels),0)
            
            fuse_inputs2 = torch.cat((inputs2, pseudo_inputs2),0)
            fuse_labels2 = torch.cat((labels2, pseudo_labels2),0)

            # ----------- domain inputs----------
            if d_target_batchsize > 0:
                src_weight = torch.FloatTensor([1.]*int(args.batch_size/2))
                tgt_weight = torch.FloatTensor([1.]*d_target_batchsize)

                dom_feat_weight = torch.cat((src_weight, d_pseudo_all_dom_conf.float(), tgt_weight),0)

                domain_inputs = torch.cat((d_inputs, d_pseudo_source_inputs, d_pseudo_all_inputs, d_target_inputs),0)

            else:
                src_weight = torch.FloatTensor([1.]*int(args.batch_size/2))

                dom_feat_weight = torch.cat((src_weight, d_pseudo_all_dom_conf.float()),0)

                domain_inputs = torch.cat((d_inputs, d_pseudo_source_inputs, d_pseudo_all_inputs),0)

            domain_labels = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                             +[0.]*int(args.batch_size / 2))

            if d_target_batchsize2 > 0:
                src_weight2 = torch.FloatTensor([1.]*int(args.batch_size/2))
                tgt_weight2 = torch.FloatTensor([1.]*d_target_batchsize2)

                dom_feat_weight2 = torch.cat((src_weight2, d_pseudo_all_dom_conf2.float(), tgt_weight2),0)

                domain_inputs2 = torch.cat((d_inputs2, d_pseudo_source_inputs2, d_pseudo_all_inputs2, d_target_inputs2),0)

            else:
                src_weight2 = torch.FloatTensor([1.]*int(args.batch_size/2))

                dom_feat_weight2 = torch.cat((src_weight2, d_pseudo_all_dom_conf2.float()),0)

                domain_inputs2 = torch.cat((d_inputs2, d_pseudo_source_inputs2, d_pseudo_all_inputs2),0)

            domain_labels2 = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                             +[0.]*int(args.batch_size / 2))

        # -------------------- train model -----------------------
        inputs, labels = Variable(fuse_inputs.cuda(0)), Variable(fuse_labels.cuda(0))
        inputs2, labels2 = Variable(fuse_inputs2.cuda(1)), Variable(fuse_labels2.cuda(1))

        domain_inputs, domain_labels = Variable(domain_inputs.cuda(0)), \
                                                     Variable(domain_labels.cuda(0))
        
        domain_inputs2, domain_labels2 = Variable(domain_inputs2.cuda(1)), \
                                                     Variable(domain_labels2.cuda(1))

        source_weight_tensor = torch.FloatTensor([1.]*source_batchsize)
        source_weight_tensor2 = torch.FloatTensor([1.]*source_batchsize2)

        if pseudo_batchsize <= 0:
            class_weights_tensor = source_weight_tensor
            class_weights_tensor2 = source_weight_tensor2
        else:
            pseudo_weights_tensor = torch.FloatTensor(pseudo_weights.float())
            pseudo_weights_tensor2 = torch.FloatTensor(pseudo_weights2.float())

            class_weights_tensor = torch.cat((source_weight_tensor, pseudo_weights_tensor),0)
            class_weights_tensor2 = torch.cat((source_weight_tensor2, pseudo_weights_tensor2),0)
        
        class_weight = Variable(class_weights_tensor.cuda(0))
        class_weight2 = Variable(class_weights_tensor2.cuda(1))

        # inputs_var = torch.autograd.Variable(inputs)
        # domain_inputs_var = torch.autograd.Variable(domain_inputs)
        # labels_var = torch.autograd.Variable(labels)
        # domain_labels_var = torch.autograd.Variable(domain_labels)

        # compute output

        p = epoch / (args.epochs * args.reverse_epoch_ratio)
        l = (2. / (1. + np.exp(-10. * p))) - 1 
        # continue
        
        # --------------------- ------------------------------- --------
        # ------------ training classification losses ------------------
        # --------------------- ------------------------------- --------

        # ----------------- classification part forward -------------------
        class_outputs = model('cls_train',input=inputs)

        class_loss = compute_new_loss(class_outputs, labels, class_weight)

        # measure accuracy and record loss
        prec1, _ = accuracy(class_outputs.data, labels.data, topk=(1,5))
        losses.update(class_loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        optimizer.zero_grad()

        class_loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # ----------------- classification part forward modality 2 -------------------

        class_outputs2 = model2('cls_train',input=inputs2)

        class_loss2 = compute_new_loss(class_outputs2, labels2, class_weight2)

        prec1_2, _ = accuracy(class_outputs2.data, labels2.data, topk=(1,5))
        losses_2.update(class_loss2.data[0], inputs2.size(0))
        top1_2.update(prec1_2[0], inputs2.size(0))

        optimizer2.zero_grad()

        class_loss2.backward()

        if args.clip_gradient is not None:
            total_norm2 = clip_grad_norm(model2.parameters(), args.clip_gradient)
            if total_norm2 > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm2, args.clip_gradient / total_norm2))

        optimizer2.step()

        # --------------------- ------------------------------- --------
        # ----------- calculate domain labels and losses ---------------
        # --------------------- ------------------------------- --------
        
        # ------------------- domain part forward ------------------------ , w_main, w_l1, w_l2, w_l3, l1_rev, l2_rev, l3_rev\ 


        domain_outputs, domain_outputs_l2, domain_outputs_l3, \
                domain_outputs_l4, w_main, w_l2, w_l3, w_l4 \
                                          = model('dom_train', input=domain_inputs, l=l)

        domain_outputs_feat, domain_outputs_l2_feat, domain_outputs_l3_feat, \
                domain_outputs_l4_feat, w_main_feat, w_l2_feat, w_l3_feat, w_l4_feat\
                        = domain_outputs, domain_outputs_l2, domain_outputs_l3, \
                                    domain_outputs_l4, w_main, w_l2, w_l3, w_l4
                        # model('dom_train', input=domain_feat_inputs, l=l)
                                    
        domain_loss = domain_criterion(domain_outputs, domain_labels)
        domain_loss_l2 = domain_criterion(domain_outputs_l2, domain_labels)
        domain_loss_l3 = domain_criterion(domain_outputs_l3, domain_labels)
        domain_loss_l4 = domain_criterion(domain_outputs_l4, domain_labels)
        
        domain_preds = torch.trunc(2*F.sigmoid(domain_outputs)).data

        domain_epoch_corrects = torch.sum(domain_preds == domain_labels.data.float())
        domain_acc = domain_epoch_corrects * 100 / domain_preds.size(0)
        domain_meter.update(domain_acc, domain_preds.size(0))

        w_main = w_main[0].expand_as(domain_loss)
        w_l2 = w_l2[0].expand_as(domain_loss_l2)
        w_l3 = w_l3[0].expand_as(domain_loss_l3)
        w_l4 = w_l4[0].expand_as(domain_loss_l4)
        
        # ------- domain classifier update ----------
        dom_loss = torch.abs(w_main)*domain_loss+ \
                   torch.abs(w_l2)*domain_loss_l2 + \
                   torch.abs(w_l3)*domain_loss_l3 + \
                   torch.abs(w_l4)*domain_loss_l4
        
        dom_optimizer.zero_grad()
        dom_loss.backward(retain_graph=True)

        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
        #     if total_norm > args.clip_gradient:
        #         print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        dom_optimizer.step()

        # ------- domain weights update ----------
        if epoch >= pre_epochs:
            dom_w_loss = w_main*domain_loss+ \
                         w_l2*domain_loss_l2+ \
                         w_l3*domain_loss_l3+ \
                         w_l4*domain_loss_l4
                
            dom_w_optimizer.zero_grad()
            dom_w_loss.backward(retain_graph=True)
            dom_w_optimizer.step()

        # --------------------- ------------------------------- --------
        # ----------------- calculate domain feat losses ---------------
        # --------------------- ------------------------------- --------
        # ---------- domain feature update ----------

        # dom_feat_weight_tensor = dom_feat_weight.cuda()

        domain_feat_criterion = nn.BCEWithLogitsLoss().cuda()
        # weight=dom_feat_weight_tensor

        domain_loss_feat = domain_feat_criterion(domain_outputs_feat, domain_labels)
        domain_loss_l2_feat = domain_feat_criterion(domain_outputs_l2_feat, domain_labels)
        domain_loss_l3_feat = domain_feat_criterion(domain_outputs_l3_feat, domain_labels)
        domain_loss_l4_feat = domain_feat_criterion(domain_outputs_l4_feat, domain_labels)

        w_main_feat = w_main_feat[0].expand_as(domain_loss_feat)
        w_l2_feat = w_l2_feat[0].expand_as(domain_loss_l2_feat)
        w_l3_feat = w_l3_feat[0].expand_as(domain_loss_l3_feat)
        w_l4_feat = w_l4_feat[0].expand_as(domain_loss_l4_feat)

        dom_feat_loss = torch.abs(w_main_feat)*domain_loss_feat+ \
                        torch.abs(w_l2_feat)*domain_loss_l2_feat+ \
                        torch.abs(w_l3_feat)*domain_loss_l3_feat+ \
                        torch.abs(w_l4_feat)*domain_loss_l4_feat


        total_epoch_loss += dom_feat_loss.data[0]
        dom_feat_optimizer.zero_grad()
        dom_feat_loss.backward()

        # if args.clip_gradient is not None:
        #     total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
        #     if total_norm > args.clip_gradient:
        #         print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        dom_feat_optimizer.step()

        # --------------------- ------------------------------- --------
        # --------------------- ------------------------------- --------
        # ------------------- domain part forward modality 2 ------------------------ 

        domain_outputs2, domain_outputs_l2_2, domain_outputs_l3_2, \
                domain_outputs_l4_2, w_main_2, w_l2_2, w_l3_2, w_l4_2 \
                                          = model2('dom_train', input=domain_inputs2, l=l)

        domain_outputs_feat2, domain_outputs_l2_feat2, domain_outputs_l3_feat2, \
                domain_outputs_l4_feat2, w_main_feat2, w_l2_feat2, w_l3_feat2, w_l4_feat2\
                        = domain_outputs2, domain_outputs_l2_2, domain_outputs_l3_2, \
                                domain_outputs_l4_2, w_main_2, w_l2_2, w_l3_2, w_l4_2

        domain_loss_2 = domain_criterion(domain_outputs2, domain_labels2)
        domain_loss_l2_2 = domain_criterion(domain_outputs_l2_2, domain_labels2)
        domain_loss_l3_2 = domain_criterion(domain_outputs_l3_2, domain_labels2)
        domain_loss_l4_2 = domain_criterion(domain_outputs_l4_2, domain_labels2)

        
        domain_preds2 = torch.trunc(2*F.sigmoid(domain_outputs2)).data

        domain_epoch_corrects2 = torch.sum(domain_preds2 == domain_labels2.data.float())
        domain_acc_2 = domain_epoch_corrects2 * 100 / domain_preds2.size(0)
        domain_meter_2.update(domain_acc_2, domain_preds2.size(0))

        w_main_2 = w_main_2[0].expand_as(domain_loss_2)
        w_l2_2 = w_l2_2[0].expand_as(domain_loss_l2_2)
        w_l3_2 = w_l3_2[0].expand_as(domain_loss_l3_2)
        w_l4_2 = w_l4_2[0].expand_as(domain_loss_l4_2)


        # ------- domain classifier update 2 ----------
        dom_loss2 = torch.abs(w_main_2)*domain_loss_2+ \
                   torch.abs(w_l2_2)*domain_loss_l2_2 + \
                   torch.abs(w_l3_2)*domain_loss_l3_2 + \
                   torch.abs(w_l4_2)*domain_loss_l4_2
        
        dom_optimizer2.zero_grad()
        dom_loss2.backward(retain_graph=True)

        dom_optimizer2.step()

       
        # ------- domain weights update 2 ----------
        if epoch >= pre_epochs:
            dom_w_loss2 = w_main_2*domain_loss_2+ \
                         w_l2_2*domain_loss_l2_2+ \
                         w_l3_2*domain_loss_l3_2+ \
                         w_l4_2*domain_loss_l4_2
                
            dom_w_optimizer2.zero_grad()
            dom_w_loss2.backward(retain_graph=True)
            dom_w_optimizer2.step()

        # --------------------- ------------------------------- --------
        # ----------------- calculate domain 2 feat losses ---------------
        # --------------------- ------------------------------- --------

        domain_loss_feat2 = domain_feat_criterion(domain_outputs_feat2, domain_labels2)
        domain_loss_l2_feat2 = domain_feat_criterion(domain_outputs_l2_feat2, domain_labels2)
        domain_loss_l3_feat2 = domain_feat_criterion(domain_outputs_l3_feat2, domain_labels2)
        domain_loss_l4_feat2 = domain_feat_criterion(domain_outputs_l4_feat2, domain_labels2)

        w_main_feat2 = w_main_feat2[0].expand_as(domain_loss_feat2)
        w_l2_feat2 = w_l2_feat2[0].expand_as(domain_loss_l2_feat2)
        w_l3_feat2 = w_l3_feat2[0].expand_as(domain_loss_l3_feat2)
        w_l4_feat2 = w_l4_feat2[0].expand_as(domain_loss_l4_feat2)


        dom_feat_loss2 = torch.abs(w_main_feat2)*domain_loss_feat2+ \
                        torch.abs(w_l2_feat2)*domain_loss_l2_feat2+ \
                        torch.abs(w_l3_feat2)*domain_loss_l3_feat2+ \
                        torch.abs(w_l4_feat2)*domain_loss_l4_feat2


        dom_feat_optimizer2.zero_grad()
        dom_feat_loss2.backward()

        dom_feat_optimizer2.step()

        # # --------------------- -------------------
                
        # # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        w2 = w_l2.data[0]
        w3 = w_l3.data[0]
        w4 = w_l4.data[0]
        wm = w_main.data[0]

        w2_2 = w_l2_2.data[0]
        w3_2 = w_l3_2.data[0]
        w4_2 = w_l4_2.data[0]
        wm_2 = w_main_2.data[0]

        epoch_acc_s = top1.avg
        epoch_acc_s2 = top1_2.avg

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec2@1 {top1_2.val:.3f} ({top1_2.avg:.3f})\t'
                  'Domain {domain_meter.val:.3f} ({domain_meter.avg:.3f})\t'
                  'Domain2 {domain_meter_2.val:.3f} ({domain_meter_2.avg:.3f})\t'
                  'W2,3,4,main: {3:.3f} {4:.3f} {5:.3f} {6:.3f}\t'
                  'W2_2,3_2,4_2,main_2: {7:.3f} {8:.3f} {9:.3f} {10:.3f}\t'
                  'l:{l:.3f}'.format(
                   epoch, i, len(source_iter), w2, w3, w4, wm, w2_2, w3_2, w4_2, wm_2, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top1_2=top1_2, domain_meter=domain_meter, domain_meter_2=domain_meter_2, lr=optimizer.param_groups[-1]['lr'], l=l))) 

        i += 1

    # ----------------------------------------------------------------
    source_acc_point.append(top1.avg)
    domain_acc_point.append(domain_meter.avg)

    source_acc_point2.append(top1_2.avg)
    domain_acc_point2.append(domain_meter_2.avg)

    if epoch < pre_epochs:
        if epoch == pre_epochs - 1:
            pre_epoch_acc_s = (pre_epoch_acc_s + epoch_acc_s) / 2
            pre_epoch_acc_s2 = (pre_epoch_acc_s2 + epoch_acc_s2) / 2
            # pre_epoch_loss_s = (pre_epoch_loss_s + epoch_loss_s) / 2
            # total_epoch_acc_s += pre_epoch_acc_s
            # train_num += 1
        else:
            pre_epoch_acc_s = epoch_acc_s
            pre_epoch_acc_s2 = epoch_acc_s2
            # pre_epoch_loss_s = epoch_loss_s

        # total_epoch_acc_s += epoch_acc_s
    else:
        train_num += 1
        total_epoch_acc_s += epoch_acc_s
        total_epoch_acc_s2 += epoch_acc_s2
        # total_epoch_loss_s += epoch_loss_s
        avg_epoch_acc_s = total_epoch_acc_s / train_num
        avg_epoch_acc_s2 = total_epoch_acc_s2 / train_num
    # avg_epoch_loss_s = total_epoch_loss_s / train_num
    print("train_num", train_num, "total_epoch_acc_s", total_epoch_acc_s)


    # ----------------------------------------------------------------
    # ------------------------ draw graph ----------------------------
    # ----------------------------------------------------------------
    try:
        os.makedirs('./graph/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    # Create plots 2
    epoch_point = [i for i in range(epoch+1)]
    fig, ax = plt.subplots()
    ax.plot(epoch_point, source_acc_point, 'k', label='Source Classification Accuracy',color='m')
    ax.plot(epoch_point, domain_acc_point, 'k', label='Domain Accuracy',color='0.6')
    ax.plot(epoch_point, target_acc_point, 'k', label='Test Classification Accuracy',color='r')

    ax.plot(epoch_point, source_acc_point2, 'k', label='Source Classification Accuracy2',color='c')
    ax.plot(epoch_point, domain_acc_point2, 'k', label='Domain Accuracy2',color='k')
    ax.plot(epoch_point, target_acc_point2, 'k', label='Test Classification Accuracy2',color='b')

    ax.annotate("ADV", xy=(1.05, 0.5), xycoords='axes fraction')
    ax.annotate('lr: %0.4f Pre epochs: %d Max epochs: %d' % (args.lr, int(args.pre_ratio * args.epochs), args.epochs), xy=(1.05, 0.65), xycoords='axes fraction')
    # ax.annotate('Pretrain epochs: %d' % PRETRAIN_EPOCH, xy=(1.05, 0.6), xycoords='axes fraction')
    # ax.annotate('Confidence Threshold: %0.3f' % confid_threshold, xy=(1.05, 0.55), xycoords='axes fraction')
    # ax.annotSate('Discriminator Threshold: %0.3f ~ %0.3f' % (LOW_DISCRIM_THRESH_T, UP_DISCRIM_THRESH_T), xy=(1.05, 0.5), xycoords='axes fraction')
    ax.annotate('1.L2,L3,L4,Main,2..: %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f' % \
                (w2, w3, w4, wm, w2_2, w3_2, w4_2, wm_2), xy=(1.05, 0.4), xycoords='axes fraction')

    if epoch >= 49:
        ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
    if epoch >= 99:
        ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
        ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
    if epoch >= 149:
        ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
        ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
        ax.annotate('150 Epoch Accuracy: %0.4f' % (target_acc_point[149]), xy=(1.05, 0.25), xycoords='axes fraction')
    if epoch >= 199:
        ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
        ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
        ax.annotate('150 Epoch Accuracy: %0.4f' % (target_acc_point[149]), xy=(1.05, 0.25), xycoords='axes fraction')
        ax.annotate('200 Epoch Accuracy: %0.4f' % (target_acc_point[199]), xy=(1.05, 0.2), xycoords='axes fraction')
    if epoch >= args.epochs:
        ax.annotate('%d Epoch Accuracy: %0.4f' % (int(args.epochs),target_acc_point[args.epochs-1]), xy=(1.05, 0.15), xycoords='axes fraction')

    ax.annotate('Mod1 Last Epoch Accuracy: %0.4f' % (target_acc_point[-1]), xy=(1.05, 0.1), xycoords='axes fraction', size=14)
    ax.annotate('Mod2 Last Epoch Accuracy: %0.4f' % (target_acc_point2[-1]), xy=(1.05, 0.05), xycoords='axes fraction', size=14)

    # Now add the legend with some customizations.
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    fig.text(0.5, 0.02, 'EPOCH', ha='center')
    fig.text(0.02, 0.5, 'ACCURACY', va='center', rotation='vertical')


    for f in glob.glob('graph/'+args.snapshot_pref+'*.png'):
        os.remove(f)


    plt.savefig('graph/'+args.snapshot_pref+'_epoch_'+str(epoch)+'_acc_{0:.3f}_{1:.3f}'.format(target_acc_point[-1], target_acc_point2[-1])+'.png', bbox_inches='tight')
    
    fig.clf()

    plt.clf()


def test(test_iter, test_iter2, test_model, test_model2, criterion, domain_criterion, iter, epoch, num_class, logger=None):
    global args

    top1_test = AverageMeter()
    top1_2_test = AverageMeter()
    domain_meter_test = AverageMeter()
    domain_meter2_test = AverageMeter()

        # ------- test model --------

    test_model.train(False)
    test_model.eval()

    Pseudo_set = []
    Select_T1_set = []
    total_pseudo_errors = 0

    # calculate target set pseudo label

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    # copy weights from model to test model


    # ---------------- test modality 2 --------------------------

    test_model2.train(False)
    test_model2.eval()

    Pseudo_set2 = []
    Select_T1_set2 = []
    total_pseudo_errors2 = 0

    if args.modality2 == 'RGB':
        length2 = 3
    elif args.modality2 == 'Flow':
        length2 = 10
    elif args.modality2 == 'RGBDiff':
        length2 = 18
    else:
        raise ValueError("Unknown modality "+args.modality2)

    # copy weights from model to test model


    # --------------- combined set ------------------

    Pseudo_set_co = []
    Select_T1_set_co = []
    total_pseudo_errors_co = 0

    # get pseudo labels for model1

    pseudo_time = time.time()

    for cnt in range(len(test_iter)):

        test_inputs, test_labels, test_path, test_num_frame = test_iter.next()
        test_inputs2, test_labels2, test_path2, test_num_frame2 = test_iter2.next()

        test_inputs = torch.autograd.Variable(test_inputs.view(-1, length, test_inputs.size(2), test_inputs.size(3)),
                                                volatile=True).cuda(0)
        test_inputs2 = torch.autograd.Variable(test_inputs2.view(-1, length2, test_inputs2.size(2), test_inputs2.size(3)),
                                                volatile=True).cuda(1)

        # test_labels = Variable(test_labels).cuda(0)
        # test_labels2 = Variable(test_labels2).cuda(1)

        domain_labels_t = Variable(torch.FloatTensor([0.]).cuda(0))
        domain_labels_t_2 = Variable(torch.FloatTensor([0.]).cuda(1))

        # -------- class 1 ------------
        class_t, domain_out_t = test_model('pseudo_discriminator', test_inputs)

        pseudo_time_out = time.time()

        dom_prob = F.sigmoid(domain_out_t.squeeze()).mean()

        class_t_avg = class_t.view(args.test_crops, args.test_segments, num_class).mean(0)

        class_t_avg_prob = F.softmax(class_t_avg, dim=1).mean(0)

        top_prob, top_label = torch.topk(class_t_avg_prob.squeeze(), 1)

        confid_rate = top_prob
        
        s_tuple = (test_path, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(test_labels[0]), test_num_frame[0])

        Pseudo_set.append(s_tuple)
        # -------- domain 1 ------------

        domain_preds_test = torch.trunc(2*F.sigmoid(domain_out_t)).data

        domain_epoch_corrects_test = torch.sum(domain_preds_test == domain_labels_t.data.float())
        domain_acc_test = domain_epoch_corrects_test * 100 / domain_preds_test.size(0)

        prec1_test = int(int(top_label[0].cpu().data[0]) == int(test_labels[0])) * 100 / top_label.size(0)

        top1_test.update(prec1_test, top_label.size(0))

        domain_meter_test.update(domain_acc_test, domain_preds_test.size(0))

        # -------- class 2 ------------
        class_t2, domain_out_t2 = test_model2('pseudo_discriminator', test_inputs2)

        dom_prob2 = F.sigmoid(domain_out_t2.squeeze()).mean()
        dom_prob_co = 0.5 * (dom_prob.data[0] + dom_prob2.data[0])

        class_t_avg2 = class_t2.view(args.test_crops, args.test_segments, num_class).mean(0)
        class_t_avg_prob2 = F.softmax(class_t_avg2, dim=1).mean(0)
        class_t_avg_prob_co = 0.5 * (class_t_avg_prob.cpu() + class_t_avg_prob2.cpu())

        top_prob2, top_label2 = torch.topk(class_t_avg_prob2.squeeze(), 1)
        top_prob_co, top_label_co = torch.topk(class_t_avg_prob_co.squeeze(), 1)

        confid_rate2 = top_prob2
        confid_rate_co = top_prob_co

        s_tuple2 = (test_path2, top_label2.data[0], confid_rate2.data[0], dom_prob2.data[0], confid_rate2.data[0], int(test_labels2[0]), test_num_frame2[0])
        s_tuple_co = (test_path, top_label_co.data[0], confid_rate_co.data[0], dom_prob_co, confid_rate_co.data[0], int(test_labels[0]), test_num_frame[0])

        Pseudo_set2.append(s_tuple2)
        Pseudo_set_co.append(s_tuple_co)
        
        # -------- domain 2 ------------
        domain_preds2_test = torch.trunc(2*F.sigmoid(domain_out_t2)).data

        prec1_2_test = int(int(top_label2[0].cpu().data[0]) == int(test_labels2[0])) * 100 / top_label2.size(0)

        top1_2_test.update(prec1_2_test, top_label2.size(0))

    test_time = time.time() - pseudo_time

    # target_acc_point.append(top1_test.avg)
    # target_acc_point2.append(top1_2_test.avg)

    # print("Pseudo error/error2/error_co/total Pseudo = {}/{} total {}".format(
    #                 total_pseudo_errors,total_pseudo_errors2,total_pseudo_errors_co,len(Pseudo_set), len(dset_loaders['test']))) 

    print('Test Epoch: [{0}]\t'
                  'Time {1} \t'
                  'Prec@1 {top1_test.avg:.3f}\t'
                  'Prec2@1 {top1_2_test.avg:.3f}\t'
                  'Domain {domain_meter_test.avg:.3f}\t'.format(
                   epoch, str(test_time),
                   top1_test=top1_test, top1_2_test=top1_2_test, domain_meter_test=domain_meter_test)) 

    return [], [], []

def validate(val_iter, val_iter2, model, model2, criterion, domain_criterion, iter, epoch, logger=None):
    global args, target_acc_point, target_acc_point2

    Pseudo_set = []
    Pseudo_set2 = []
    Pseudo_set_co = []

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_2 = AverageMeter()
    domain_meter = AverageMeter()
    domain_meter2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model2.eval()

    start_time = time.time()

    # for i, (input, target, _, _) in enumerate(val_loader):
    for cnt in range(len(val_iter)):

        input, target, path, num_frame = val_iter.next()
        input2, target2, path2, num_frame2 = val_iter2.next()

        # if i > args.evalBreakIter:
        #     break
        target = target.cuda(0)
        target2 = target2.cuda(1)

        input_var = torch.autograd.Variable(input, volatile=True).cuda(0)
        input_var2 = torch.autograd.Variable(input2, volatile=True).cuda(1)

        target_var = torch.autograd.Variable(target, volatile=True).cuda(0)
        target_var2 = torch.autograd.Variable(target2, volatile=True).cuda(1)

        # compute output
        output, domain_out = model('pseudo_discriminator', input_var)
        output2, domain_out2= model2('pseudo_discriminator', input_var2)

        loss = criterion(output, target_var)
        loss2 = criterion(output2, target_var2)

        # ----- pseudo labels -----------

        dom_prob = F.sigmoid(domain_out.squeeze()).mean()
        dom_prob2 = F.sigmoid(domain_out2.squeeze()).mean()
        dom_prob_co = 0.5 * (dom_prob.data[0] + dom_prob2.data[0])

        class_t_avg_prob = F.softmax(output, dim=1)
        class_t_avg_prob2 = F.softmax(output2, dim=1)
        class_t_avg_prob_co = 0.5 * (class_t_avg_prob.cpu() + class_t_avg_prob2.cpu())

        top_prob, top_label = torch.topk(class_t_avg_prob.squeeze(), 1)
        top_prob2, top_label2 = torch.topk(class_t_avg_prob2.squeeze(), 1)
        top_prob_co, top_label_co = torch.topk(class_t_avg_prob_co.squeeze(), 1)

        confid_rate = top_prob
        confid_rate2 = top_prob2
        confid_rate_co = top_prob_co

        s_tuple = (path, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(target[0]), num_frame[0])
        s_tuple2 = (path2, top_label2.data[0], confid_rate2.data[0], dom_prob2.data[0], confid_rate2.data[0], int(target2[0]), num_frame2[0])
        s_tuple_co = (path, top_label_co.data[0], confid_rate_co.data[0], dom_prob_co, confid_rate_co.data[0], int(target[0]), num_frame[0])

        Pseudo_set.append(s_tuple)
        Pseudo_set2.append(s_tuple2)
        Pseudo_set_co.append(s_tuple_co)

        # measure accuracy and record loss
        prec1 = int(int(top_label[0].cpu().data[0]) == int(target[0])) * 100 / top_label.size(0)
        prec1_2 = int(int(top_label2[0].cpu().data[0]) == int(target2[0])) * 100 / top_label2.size(0)

        # losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        top1_2.update(prec1_2, input2.size(0))

        # --------- domain acc -------------
        domain_preds = torch.trunc(2*dom_prob).data
        domain_preds2 = torch.trunc(2*dom_prob2).data

        domain_labels = Variable(torch.FloatTensor([0.]).cuda(0))
        domain_labels2 = Variable(torch.FloatTensor([0.]).cuda(1))

        domain_epoch_corrects = torch.sum(domain_preds == domain_labels.data.float())
        domain_epoch_corrects2 = torch.sum(domain_preds2 == domain_labels2.data.float())
        domain_acc = domain_epoch_corrects * 100 / domain_preds.size(0)
        domain_acc2 = domain_epoch_corrects2 * 100 / domain_preds2.size(0)

        domain_meter.update(domain_acc, domain_preds.size(0))
        domain_meter2.update(domain_acc2, domain_preds2.size(0))

    val_time = time.time() - start_time

    target_acc_point.append(top1.avg)
    target_acc_point2.append(top1_2.avg)

    print('Val Epoch: [{0}]\t'
                  'Time {1} \t'
                  'Prec@1 {top1.avg:.3f}\t'
                  'Prec2@1 {top1_2.avg:.3f}\t'
                  'Domain {domain_meter.avg:.3f}\t'
                  'Domain2 {domain_meter2.avg:.3f}\t'.format(
                   epoch, str(val_time),
                   top1=top1, top1_2=top1_2, domain_meter=domain_meter, domain_meter2=domain_meter2)) 

    return top1.avg, top1_2.avg, Pseudo_set, Pseudo_set2, Pseudo_set_co
    
def pull_back_to_n(x, n):
    if x > n:
        x = n
    return x

def save_checkpoint(state, epoch=0, mod="", filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, mod, filename))
    torch.save(state, filename)

    if epoch!=0:
        filename_e = '_'.join((args.snapshot_pref, mod, "Epoch", str(epoch), 'checkpoint.pth.tar'))
        torch.save(state, filename_e)
    # if is_best:
    #     best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
    #     shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch, lr_steps):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
#     lr = args.lr * decay
#     decay = args.weight_decay
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr * param_group['lr_mult']
#         param_group['weight_decay'] = decay * param_group['decay_mult']

def adjust_learning_rate(optimizer, lr_arg, decay):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = lr_arg * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

# def lr_scheduler(optimizer, lr_mult, weight_mult=1):
#     counter = 0
#     for param_group in optimizer.param_groups:
#         if counter == 0:
#             optimizer.param_groups[counter]['lr'] = args.lr * lr_mult / 10.0
#         else:
#             optimizer.param_groups[counter]['lr'] = args.lr * lr_mult
#         counter += 1

#     return optimizer, lr_mult

def dom_w_scheduler(optimizer, decay, weight_mult=1):
    counter = 0
    for param_group in optimizer.param_groups:
        if counter == 0:
            param_group[counter]['lr'] = args.lr * decay * weight_mult
            param_group['weight_decay'] = decay * param_group['decay_mult']
        counter += 1

    return optimizer, decay

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def pull_back_to_one(x):
    if x > 1:
        x = 1
    return x

def compute_new_loss(logits, target, weights):
    """ logits: Unnormalized probability for each class.
        target: index of the true class(label)
        weights: weights of weighted loss.
    Returns:
        loss: An average weighted loss value
    """
    # print("l: ",logits)
    # print("t: ",target)
    weights = weights.narrow(0,0,len(target))
    # print("w: ",weights)
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size()) * weights
    # losses = losses * weights
    loss = losses.sum() / len(target)
    # length.float().sum()
    return loss

if __name__ == '__main__':
    main()
