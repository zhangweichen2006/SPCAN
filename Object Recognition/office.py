from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.model_zoo as model_zoo

import matplotlib
from matplotlib.offsetbox import AnchoredText
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time, math
import copy
import os,errno
# import bisect
from operator import itemgetter
from discriminative_dann import *
import imagefolder
import argparse
from PIL import Image, ImageDraw,ImageFont

######################################################################
# Load Data

parser = argparse.ArgumentParser(description='Domain Adaptation')
parser.add_argument('--source_set', type=str, default='amazon')
parser.add_argument('--target_set', type=str, default='webcam')

parser.add_argument('--gpu', type=str, default='2')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_class', type=int, default=31)
parser.add_argument('--base_lr', type=float, default=0.0015)

parser.add_argument('--pretrain_sample', type=int, default=50000)
parser.add_argument('--train_sample', type=int, default=200000)

parser.add_argument('--form_w', type=float, default=0.4)
parser.add_argument('--main_w', type=float, default=-0.8)

parser.add_argument('--wp', type=float, default=0.055)
parser.add_argument('--wt', type=float, default=1)

parser.add_argument('--select', type=str, default='1-2')

parser.add_argument('--usePreT2D', type=bool, default=False)

parser.add_argument('--useT1DorT2', type=str, default="T2")

parser.add_argument('--diffS', type=bool, default=False)

parser.add_argument('--diffDFT2', type=bool, default=False)

parser.add_argument('--useT2CompD', type=bool, default=False)
parser.add_argument('--usemin', type=bool, default=False)

parser.add_argument('--useRatio', type=bool, default=False)

parser.add_argument('--useCurrentIter', type=bool, default=False)
# parser.add_argument('--useEpoch', type=bool, default=False)

parser.add_argument('--useLargeLREpoch', type=bool, default=True)

parser.add_argument('--MaxStep', type=int, default=0)

parser.add_argument('--useSepTrain', type=bool, default=True)

parser.add_argument('--fixW', type=bool, default=False)
parser.add_argument('--decay', type=float, default=0.0003)
parser.add_argument('--nesterov', type=bool, default=True)

parser.add_argument('--ReTestSource', type=bool, default=False)

parser.add_argument('--sourceTestIter', type=int, default=2000)
parser.add_argument('--defaultPseudoRatio', type=float, default=0.2)
parser.add_argument('--totalPseudoChange', type=int, default=100)

parser.add_argument('--beta', type=float, default=1.0)

args = parser.parse_args()

data_dir = '/home/wzha8158/datasets/Office/domain_adaptation_images/'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

######################################################################

print('OFFICE-31: ' + args.source_set + ' To ' + args.target_set)

data_transforms = {
    'source': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'target': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dsets = {}
dsets['source'] = imagefolder.ImageFolder(data_dir+args.source_set+'/images', data_transforms['source'])
dsets['d_source'] = imagefolder.ImageFolder(data_dir+args.source_set+'/images', data_transforms['source'])

dsets['test_source'] = imagefolder.ImageFolder(data_dir+args.source_set+'/images', data_transforms['test'])
# dsets['target_source'] = imagefolder.ImageFolder(data_dir+args.source_set, data_transforms['source'])

dsets['target'] = imagefolder.ImageFolder(data_dir+args.target_set+'/images', data_transforms['target'])
dsets['target_test'] = imagefolder.ImageFolder(data_dir+args.target_set+'/images', data_transforms['test'])

dsets['d_pseudo_source'] = imagefolder.ImageFolder(data_dir+args.source_set+'/images', data_transforms['source'])

dsets['pseudo'] = []
dsets['d_pseudo_target'] = []

dsets['d_pseudo_source_feat'] = []
dsets['d_pseudo_target_feat'] = []

dsets['d_pseudo_target_comp'] = []
dsets['d_pseudo_target_inter'] = []
dsets['d_pseudo_target_rev'] = []
dsets['d_pseudo_target_rev_feat'] = []

dsets['d_pseudo'] = []
dsets['d_pseudo_feat'] = []

dsets['d_pseudo_all'] = []
dsets['d_pseudo_all_feat'] = [] 

dsets['test'] = imagefolder.ImageFolder(data_dir+args.target_set+'/images', data_transforms['test'])
dsets['draw'] = imagefolder.ImageFolder(data_dir+args.target_set+'/images', data_transforms['test'])

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=int(args.batch_size / 2),
                shuffle=True) for x in ['source', 'd_source', 'test_source', 'target', 'target_test', 'd_pseudo_source']}
# , 'test'
dset_loaders['test'] = torch.utils.data.DataLoader(dsets['test'], batch_size=1,
                shuffle=False)

source_batches_per_epoch = np.floor(len(dsets['source']) * 2 / args.batch_size).astype(np.int16)
target_batches_per_epoch = np.floor(len(dsets['target']) * 2 / args.batch_size).astype(np.int16)

pre_epochs = int(args.pretrain_sample / len(dsets['source']))
total_epochs = int(pre_epochs + args.train_sample / len(dsets['source']))

# pre_epochs = int(args.pretrain_sample / min(len(dsets['source']), len(dsets['target'])))
# total_epochs = int(pre_epochs + args.train_sample / min(len(dsets['source']), len(dsets['target'])))

pre_iters = int(args.pretrain_sample * 2 / args.batch_size)
total_iters = int(pre_iters + args.train_sample * 2 / args.batch_size)

if args.useLargeLREpoch:
    lr_epochs = 10000
else:
    lr_epochs = total_epochs
######################################################################
# Finetuning the convnet
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
                'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',}

#pseudo_dataset: dsets['target_test']
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

def train_model(model, optimizer, dom_optimizer, dom_w_optimizer, dom_feat_optimizer, cls_lr_scheduler, dom_w_lr_scheduler, feature_params, num_epochs=500):
    since = time.time()

    # ----- initialise variables ----

    double_desc = False

    best_model = model
    best_acc = 0.0
    epoch_acc_s = 0.0
    epoch_acc_t = 0.0
    epoch_loss_s = 0.0
    pre_epoch_acc_s = 0.0
    pre_epoch_loss_s = 0.0
    total_epoch_acc_s = 0.0
    total_epoch_loss_s = 0.0

    avg_epoch_acc_s = 0.0
    avg_epoch_loss_s = 0.0
    total_threshold = 0.0
    avg_threshold = 0.1
    epoch_lr_mult = 0.0
    threshold_count = 0
    threshold_list = []

    test_source_count = 0
    avg_test_source_acc = 0.0
    total_test_source_acc = 0.0

    source_step_count = 0
    target_step_count = 0

    iters = 0
    current_iters = 0

    class_loss_point = []
    domain_loss_point = []

    source_acc_point = []
    domain_acc_point = []

    target_loss_point = []
    target_acc_point = []

    set_len_point = []

    confid_threshold_point = []

    epoch_point = []
    lr_point = []

    domain_loss_point_l1 = []
    domain_loss_point_l2 = []
    domain_loss_point_l3 = []

    domain_acc_point_l1 = []
    domain_acc_point_l2 = []
    domain_acc_point_l3 = []

    # ------------------------- two model train ----------------------------------

    for epoch in range(lr_epochs):

        print('Epoch {}/{}/{}'.format(epoch, num_epochs - 1, lr_epochs))
        print('-' * 10)

        epoch_point = [i for i in range(epoch+1)]

        # ----------------------------------------------------------
        # --------------- Training and Testing Phase ---------------
        # ----------------------------------------------------------

        for phase in ['train', 'test']:

            # ----- initialise common variables -----
            epoch_loss = 0.0
            epoch_corrects = 0
            
            test_corrects = {}
            test_totals = {}
            # ----------------------------------------------------------
            # ------------------ Training Phase ------------------------
            # ----------------------------------------------------------

            if phase == 'train':

                # ----- initialise common variables -----

                domain_epoch_loss = 0.0
                total_epoch_loss = 0.0
                domain_epoch_corrects = 0
                epoch_discrim_bias = 0.0

                # ---- classifier -----
                source_pointer = 0
                pseudo_pointer = 0

                total_source_pointer = 0
                total_pseudo_pointer = 0

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

                batch_count = 0
                class_count = 0
                domain_counts = 0

                # total_iters = 0

                domain_epoch_loss_l1 = 0.0
                domain_epoch_loss_l2 = 0.0
                domain_epoch_loss_l3 = 0.0

                domain_epoch_corrects_l1 = 0
                domain_epoch_corrects_l2 = 0
                domain_epoch_corrects_l3 = 0

                w_main = 0.0 
                w_l1 = 0.0 
                w_l2 = 0.0 
                w_l3 = 0.0 

                confid_threshold = 0

                # -----------------------------------------------
                # ----------------- Pre-train -------------------
                # -----------------------------------------------

                if epoch >= pre_epochs:
                    
                    # -----------------------------------------------
                    # ------------ Pseudo Labelling -----------------
                    # -----------------------------------------------

                    # ------- with iterative pseudo sample filter --------

                    model.train(False)
                    model.eval()

                    Pseudo_set = []
                    Select_T1_set = []
                    total_pseudo_errors = 0

                    # calculate target set pseudo label

                    # Pseudo_set = cal_pseudo_label_set(model, 'target_test')
                    dset_loaders['target_test'] = torch.utils.data.DataLoader(dsets['target_test'],
                                                        batch_size=1, shuffle=False)
            
                    pseudo_counter = 0
            
                    for test_inputs, test_labels, test_path, _ in dset_loaders['target_test']:

                        test_inputs = Variable(test_inputs.cuda())

                        domain_labels_t = Variable(torch.FloatTensor([0.]*len(test_inputs)).cuda())

                        ini_weight = Variable(torch.FloatTensor([1.0]*len(test_inputs)).cuda())

                        class_t, domain_out_t, confid_rate = model('pseudo_discriminator', test_inputs,[],[],domain_labels_t,ini_weight)

                        # prediction confidence weight

                        # domain variance

                        dom_prob = F.sigmoid(domain_out_t.squeeze())

                        top_prob, top_label = torch.topk(F.softmax(class_t.squeeze()), 1)

                        # dom_confid = 1 - dom_prob.data[0]

                        # s_tuple = (test_path, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(test_labels[0]))
                        s_tuple = (test_path, top_label.data[0], confid_rate.data[0], dom_prob.data[0], confid_rate.data[0], int(test_labels[0]))
                        Pseudo_set.append(s_tuple)
                        # -------- sort domain variance score, reduce memory ------------
                        fake_sample = int(int(top_label[0].cpu().data[0]) != int(test_labels[0]))

                        total_pseudo_errors += fake_sample
                        pseudo_counter += 1

                    # ----------------- sort pseudo datasets -------------------

                    pseudo_num = len(Pseudo_set)
                    Sorted_confid_set = sorted(Pseudo_set, key=lambda tup: tup[2], reverse=True)


                    # ----------------- calculate pseudo threshold -------------------
                    threshold_count += 1

                    if args.ReTestSource:
                        current_test_source_acc = cal_test_source_accuracy(model, 'test_source')
                        max_threshold = args.totalPseudoChange
                    else:
                        current_test_source_acc = epoch_acc_s
                        avg_test_source_acc = avg_epoch_acc_s
                        max_threshold = total_epochs

                    if current_test_source_acc < avg_test_source_acc:
                        if not double_desc:
                            double_desc = True
                        else:
                            threshold_count -= 2
                    else:
                        double_desc = False
                    

                    # sourceTestIter
                    # totalPseudoChange

                    pseu_n = args.defaultPseudoRatio + threshold_count / max_threshold
                    pseu_n = pull_back_to_one(pseu_n)

                    print("threshold_count: {} pseudo_ratio:{:.4f}".format(threshold_count, pseu_n))

                    t_p = pseu_n

                    # ----------------- calculate avg test source acc -------------------
                    if args.ReTestSource:
                        test_source_count += 1
                        total_test_source_acc += current_test_source_acc
                        avg_test_source_acc = total_test_source_acc / test_source_count

                    # ----------------- get pseudo_C set -------------------
                    confid_pseudo_num = int(t_p*pseudo_num)
                    Select_confid_set = Sorted_confid_set[:confid_pseudo_num]

                    if len(Select_confid_set) > 0:
                        confid_threshold = Select_confid_set[-1][2]

                    Select_T1_set = Select_confid_set

                    # ----------------- get pseudo_D set (T1 + T2)--------------
                    Sorted_dom_set = sorted(Pseudo_set, key=lambda tup: abs(tup[3] - 0.5), reverse=True)
                    
                    t_d = pseu_n

                    domain_threshold_num = int(t_d*pseudo_num)
                    Select_T2_set = Sorted_dom_set[:domain_threshold_num] # near 0 and 1

                    Selected_T2_Minus_T1_set = [(i[0], i[1], i[2], i[3], i[4], i[5]) \
                                           for i in Select_T2_set if i[2] < confid_threshold] # T2 - T1

                    Select_T1T2_set = Select_T1_set + Selected_T2_Minus_T1_set # T1 Union T2

                    # -----------------(T1 Intersect T2)-------------------

                    # Selected_T1_Inter_T2_set = [(i[0], i[1], i[2], i[3], i[4], i[5]) \
                    #                        for i in Select_T2_set if i[2] >= confid_threshold] # T1 Intersect T2

                    # ----------------------(T1 + T3)----------------------
                    # if t_d > 0:
                    #     domain_rev_threshold_num = int((1-t_d)*pseudo_num)
                    #     Select_T3_set = Sorted_dom_set[domain_rev_threshold_num:] # near 0.5
     
                    #     Selected_T3_Minus_T1_set = [(i[0], i[1], i[2], i[3], i[4], i[5]) \
                    #                             for i in Select_T3_set if i[2] < confid_threshold]  # T3 - T1

                    # Select_T1T3_set = Select_T1_set + Selected_T3_Minus_T1_set # T1 Union T3

                    # -----------------(T1 Intersect T3)-------------------

                    # Selected_T1_Inter_T3_set = [(i[0], i[1], i[2], i[3], i[4], i[5]) \
                    #                        for i in Select_T3_set if i[2] >= confid_threshold] # T1 Intersect T3

                    total_errors = 0
                    for ss in Select_T1_set:
                        fake_sample = int(ss[1] != ss[-1])
                        total_errors += fake_sample 

                    print("Select error/total selected = {}/{} total {}, confid_threshold: {:.4f}".format(
                                    total_errors,len(Select_T1_set), len(Pseudo_set), t_p)) 
                    
                    # ----------------- get pseudo datasets -------------------
                    if len(Select_T1_set) > 0:
                        dsets['pseudo'] = imagefolder.PathFolder(Select_T1_set, data_transforms['target'])
                    if len(Select_T2_set) > 0:
                        dsets['d_pseudo_target'] = imagefolder.PathFolder(Select_T2_set, data_transforms['target'])

                    if len(Select_T2_set) > 0:
                        dsets['d_pseudo_target_feat'] = imagefolder.PathFolder(Select_T2_set, data_transforms['target'])
                        
                    # if len(Select_T3_set) > 0:
                    #     dsets['d_pseudo_target_feat'] = imagefolder.PathFolder(Select_T3_set, data_transforms['target'])

                    if len(Select_T1T2_set) > 0:
                        dsets['d_pseudo_all'] = imagefolder.PathFolder(Select_T1T2_set, data_transforms['target'])
                        dsets['d_pseudo_all_feat'] = imagefolder.PathFolder(Select_T1T2_set, data_transforms['target'])
                    # if len(Select_T1T3_set) > 0:
                    #     dsets['d_pseudo_all_feat'] = imagefolder.PathFolder(Select_T1T3_set, data_transforms['target'])

                    # if len(Selected_T1_Inter_T2_set) > 0:
                    #     dsets['d_pseudo_target_inter'] = imagefolder.PathFolder(Selected_T1_Inter_T2_set, data_transforms['target'])
                    # if len(Selected_T1_Inter_T3_set) > 0:
                    #     dsets['d_pseudo_target_inter_feat'] = imagefolder.PathFolder(Selected_T1_Inter_T3_set, data_transforms['target'])

                    # ----------------- reload pseudo set ---------------------------

                    confid_threshold_point.append(float("%.4f" % confid_threshold))

                # ---------------------------------------------------------
                # -------- Pseudo + Source Classifier Training ------------
                # ---------------------------------------------------------

                # -------- loop through source dataset ---------

                model.train(True) # Set model to training mode

                for param in model.parameters():
                    param.requires_grad = True

                for param in model.disc_activate.parameters():
                    param.requires_grad = False

                if epoch < pre_epochs:
                    for param in model.disc_weight.parameters():
                        param.requires_grad = False

                ini_w_main = Variable(torch.FloatTensor([float(args.main_w)]).cuda())
                ini_w_l1 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda())
                ini_w_l2 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda())
                ini_w_l3 = Variable(torch.FloatTensor([float(args.form_w/3)]).cuda())

                # ---------------------------------------------------------------------
                # ------- Source + Pseudo + Target Dataset Preparation ---------
                # ---------------------------------------------------------------------

                source_size = len(dsets['source'])
                pseudo_size = len(dsets['pseudo'])
                d_target_size = len(dsets['target'])
                d_pseudo_all_size = len(dsets['d_pseudo_all'])

                if pseudo_size == 0:
                    source_batchsize = int(args.batch_size / 2) 
                    pseudo_batchsize = 0
                    d_source_batchsize = int(args.batch_size / 2)
                    d_target_batchsize = int(args.batch_size / 2)
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

                    pseudo_iter = iterator_reset('pseudo', 'pseudo', pseudo_batchsize)

                    d_pseudo_source_iter = iterator_reset('d_pseudo_source', 'd_pseudo_source', d_pseudo_source_batchsize)
                    d_pseudo_source_feat_iter = iterator_reset('d_pseudo_source_feat', 'd_pseudo_source', d_pseudo_source_batchsize)

                    if d_pseudo_all_size > 0:

                        d_pseudo_all_iter = iterator_reset('d_pseudo_all', 'd_pseudo_all', d_pseudo_all_batchsize)
                        d_pseudo_all_feat_iter = iterator_reset('d_pseudo_all_feat', 'd_pseudo_all_feat', d_pseudo_all_batchsize)


                source_iter = iterator_reset('source','source',source_batchsize)

                d_source_iter = iterator_reset('d_source','source',d_source_batchsize)
                d_source_feat_iter = iterator_reset('d_source_feat','source',d_source_batchsize)

                d_target_iter = iterator_reset('d_target','target',d_target_batchsize)
                d_target_feat_iter = iterator_reset('d_target_feat','target',d_target_batchsize)

                # ---------------------------------------------------------------------
                # --------------------------- start training --------------------------
                # ---------------------------------------------------------------------

                while source_pointer < len(source_iter):
                    source_step_count +=1
                            
                    if args.useLargeLREpoch:
                        # base on decay
                        p = epoch / total_epochs
                        if args.MaxStep != 0:
                            p = source_step_count / args.MaxStep
                            
                        p = pull_back_to_one(p)
                        l = (2. / (1. + np.exp(-10. * p))) - 1
                        step_rate = args.decay * source_step_count
                        if (epoch == 0):
                            lr_mult = 1 / (1 + np.exp(-3*(source_step_count / len(dsets['source']))))
                        else:
                            lr_mult = (1. + step_rate)**(-0.75)
                        weight_mult = args.wt * args.wp **p

                    else:
                        # base on total epoch
                        p = epoch / total_epochs
                        l = (2. / (1. + np.exp(-10. * p))) - 1                        
                        if (epoch == 0):
                            lr_mult = 1 / (1 + np.exp(-3*(source_step_count / len(dsets['source']))))
                        else:
                            lr_mult = (1. + 10 * p)**(-0.75)

                    optimizer, epoch_lr_mult = cls_lr_scheduler(optimizer, lr_mult)
                    dom_optimizer, epoch_lr_mult = cls_lr_scheduler(dom_optimizer, lr_mult)
                    dom_w_optimizer, epoch_lr_mult = dom_w_lr_scheduler(dom_w_optimizer, lr_mult, weight_mult)
                    dom_feat_optimizer, epoch_lr_mult = cls_lr_scheduler(dom_feat_optimizer, lr_mult)

                    # make sure to skip the last batch if the batch length is not enough(drop last)
                    batch_count += 1
                    if (batch_count * source_batchsize > len(dsets['source'])):
                        continue
                    # --------------------- ------------------------- -----------------------
                    # --------------------- classification part batch -----------------------
                    # --------------------- ------------------------- -----------------------
                    if epoch < pre_epochs:
                        source_batchsize = int(args.batch_size / 2)
                        pseudo_batchsize = 0

                    else:
                        source_size = len(dsets['source'])
                        pseudo_size = len(dsets['pseudo']) * args.beta

                        source_batchsize = int(int(args.batch_size / 2) * source_size
                                                        / (source_size + pseudo_size))

                        if pseudo_size > 0:
                            if source_batchsize < int(int(args.batch_size / 2) / 2):
                                source_batchsize = int(int(args.batch_size / 2) / 2)

                            if source_batchsize == int(args.batch_size / 2) and epoch >= pre_epochs:
                                source_batchsize -= 1

                        pseudo_batchsize = int(args.batch_size / 2) - source_batchsize

                    # ----------------- get classification input --------------------------

                    source_iter, inputs, labels, source_pointer, total_source_pointer \
                                    = iterator_update(source_iter, 'source', source_batchsize, 
                                                      source_pointer, total_source_pointer, "ori")

                    if pseudo_batchsize > 0:
                        # if args.useRatio:
                        #     if pseudo_batchsize > 0:
                        #         if total_pseudo_pointer <= pseudo_source_ratio * (total_source_pointer + total_pseudo_pointer):
                        #             tmp_pseudo_batchsize = pseudo_batchsize
                        #         # ------------- -1 on pseduo batchsize ------------------
                        #         elif pseudo_batchsize > 1 and total_pseudo_pointer-1 <= pseudo_source_ratio * (total_source_pointer + total_pseudo_pointer):
                        #             tmp_pseudo_batchsize = pseudo_batchsize - 1
                        #         else:
                        #             tmp_pseudo_batchsize = 0
                        #     else:
                        #         tmp_pseudo_batchsize = 0
                        # else:
                        #     tmp_pseudo_batchsize = pseudo_batchsize

                        pseudo_iter, pseudo_inputs, pseudo_labels, pseudo_pointer, total_pseudo_pointer, pseudo_weights, pseudo_dom_conf \
                                        = iterator_update(pseudo_iter, 'pseudo', 
                                                           pseudo_batchsize, pseudo_pointer, 
                                                           total_pseudo_pointer, "pseu")

                    # --------------------- ------------------------------- -----------------------
                    # --------------------- domain discriminator part batch -----------------------
                    # --------------------- ------------------------------- -----------------------
                    if epoch < pre_epochs:
                        d_pseudo_source_batchsize = 0
                        d_source_batchsize = int(args.batch_size / 2)
                        d_target_batchsize = int(args.batch_size / 2)

                    else:
                        d_source_size = len(dsets['source'])
                        d_pseudo_size = len(dsets['pseudo']) * args.beta
                        d_target_size = len(dsets['target'])
                        d_pseudo_all_size = len(dsets['d_pseudo_all'])

                        d_source_batchsize = source_batchsize
                        d_pseudo_source_batchsize = pseudo_batchsize

                        d_pseudo_all_batchsize = 0
                        if d_pseudo_all_size > 0:
                            d_pseudo_all_batchsize = int(round(int(args.batch_size / 2) * d_pseudo_all_size / d_target_size))

                            if d_pseudo_all_batchsize == 0:
                                d_pseudo_all_batchsize = 1

                        d_target_batchsize = int(args.batch_size / 2) - d_pseudo_all_batchsize

                    # ----------------- get domain discriminator input --------------------------

                    d_source_iter, d_inputs, _, d_source_pointer, total_d_source_pointer \
                                    = iterator_update(d_source_iter, 'source', d_source_batchsize, 
                                                      d_source_pointer, total_d_source_pointer, "ori")

                    d_source_feat_iter, d_feat_inputs, _, d_source_feat_pointer, total_d_source_feat_pointer \
                                    = iterator_update(d_source_feat_iter, 'source', d_source_batchsize, 
                                                      d_source_feat_pointer, total_d_source_feat_pointer, "ori")

                    if d_target_batchsize > 0:
                        d_target_iter, d_target_inputs, _, d_target_pointer, total_d_target_pointer \
                                        = iterator_update(d_target_iter, 'target', d_target_batchsize, 
                                                          d_target_pointer, total_d_target_pointer, "ori")

                        d_target_feat_iter, d_target_feat_inputs, _, d_target_feat_pointer, total_d_target_feat_pointer \
                                        = iterator_update(d_target_feat_iter, 'target', d_target_batchsize, 
                                                          d_target_feat_pointer, total_d_target_feat_pointer, "ori")


                    # ----------------- get domain pseudo input --------------------------
                    if d_pseudo_source_batchsize > 0:

                        d_pseudo_source_iter, d_pseudo_source_inputs, _, d_pseudo_source_pointer, total_d_pseudo_source_pointer\
                                    = iterator_update(d_pseudo_source_iter, 'd_pseudo_source', 
                                                       d_pseudo_source_batchsize, d_pseudo_source_pointer, 
                                                       total_d_pseudo_source_pointer, "ori")
                        
                        d_pseudo_source_feat_iter, d_pseudo_source_feat_inputs, _, d_pseudo_source_feat_pointer, total_d_pseudo_source_feat_pointer \
                                    = iterator_update(d_pseudo_source_feat_iter, 'd_pseudo_source', 
                                                       d_pseudo_source_batchsize, d_pseudo_source_feat_pointer, 
                                                       total_d_pseudo_source_feat_pointer, "ori")
                        
                        # ------------------------ T1+T2 --------------------
                        d_pseudo_all_iter, d_pseudo_all_inputs, _, d_pseudo_all_pointer, total_d_pseudo_all_pointer, _, d_pseudo_all_dom_conf \
                                = iterator_update(d_pseudo_all_iter, 'd_pseudo_all', 
                                                   d_pseudo_all_batchsize, d_pseudo_all_pointer, 
                                                   total_d_pseudo_all_pointer, "pseu")

                        # ------------------------ T1+T3 --------------------
                        d_pseudo_all_feat_iter, d_pseudo_all_feat_inputs, _, d_pseudo_all_feat_pointer, total_d_pseudo_all_feat_pointer, _, d_pseudo_all_feat_dom_conf \
                                = iterator_update(d_pseudo_all_feat_iter, 'd_pseudo_all_feat', 
                                                   d_pseudo_all_batchsize, d_pseudo_all_feat_pointer, 
                                                   total_d_pseudo_all_feat_pointer, "pseu")

                    # --------------------- ------------------------------- -----------------------
                    # ----------------------------- fit model ------------- -----------------------
                    # --------------------- ------------------------------- -----------------------

                    if epoch < pre_epochs or pseudo_batchsize <= 0:
                        # ----------- classifier inputs----------
                        fuse_inputs = inputs
                        fuse_labels = labels

                        # ----------- domain inputs----------
                        domain_inputs = torch.cat((d_inputs, d_target_inputs),0)
                        domain_feat_inputs = torch.cat((d_feat_inputs, d_target_feat_inputs),0)

                        domain_labels = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                                         +[0.]*int(args.batch_size / 2))

                        dom_feat_weight = torch.FloatTensor([1.]*int(args.batch_size))

                    else:
                        # ----------- classifier inputs----------
                        fuse_inputs = torch.cat((inputs, pseudo_inputs),0)
                        fuse_labels = torch.cat((labels, pseudo_labels),0)

                        # ----------- domain inputs----------
                        if d_target_batchsize > 0:
                            src_weight = torch.FloatTensor([1.]*int(args.batch_size/2))
                            tgt_weight = torch.FloatTensor([1.]*d_target_batchsize)

                            dom_feat_weight = torch.cat((src_weight, d_pseudo_all_dom_conf.float(), tgt_weight),0)

                            domain_inputs = torch.cat((d_inputs, d_pseudo_source_inputs, d_pseudo_all_inputs, d_target_inputs),0)
                            domain_feat_inputs = torch.cat((d_feat_inputs, d_pseudo_source_feat_inputs, d_pseudo_all_feat_inputs, d_target_feat_inputs),0)

                        else:
                            src_weight = torch.FloatTensor([1.]*int(args.batch_size/2))

                            dom_feat_weight = torch.cat((src_weight, d_pseudo_all_dom_conf.float()),0)

                            domain_inputs = torch.cat((d_inputs, d_pseudo_source_inputs, d_pseudo_all_inputs),0)
                            domain_feat_inputs = torch.cat((d_feat_inputs, d_pseudo_source_feat_inputs, d_pseudo_all_feat_inputs),0)

                        domain_labels = torch.FloatTensor([1.]*int(args.batch_size / 2)
                                                         +[0.]*int(args.batch_size / 2))

                    # -------------------- train model -----------------------
                    inputs, labels = Variable(fuse_inputs.cuda()), Variable(fuse_labels.cuda())

                    domain_inputs, domain_feat_inputs, domain_labels = Variable(domain_inputs.cuda()), \
                                                                      Variable(domain_feat_inputs.cuda()),\
                                                                      Variable(domain_labels.cuda())
                    
                    source_weight_tensor = torch.FloatTensor([1.]*source_batchsize)

                    if pseudo_batchsize <= 0:
                        class_weights_tensor = source_weight_tensor
                    else:
                        pseudo_weights_tensor = torch.FloatTensor(pseudo_weights.float())
                        class_weights_tensor = torch.cat((source_weight_tensor, pseudo_weights_tensor),0)
                    
                    class_weight = Variable(class_weights_tensor.cuda())

                    # --------------------- ------------------------------- --------
                    # ------------ training classification losses ------------------
                    # --------------------- ------------------------------- --------
                    
                    # --------------------------- classification part forward ------------------------
                    class_outputs = model('cls_train', x1=inputs)
                    
                    criterion = nn.CrossEntropyLoss()

                    _, preds = torch.max(class_outputs.data, 1)
                    class_count += len(preds)
                    class_loss = compute_new_loss(class_outputs, labels, class_weight)

                    epoch_loss += class_loss.data[0]
                    total_epoch_loss += class_loss.data[0]
                    epoch_corrects += torch.sum(preds == labels.data)

                    optimizer.zero_grad()
                    class_loss.backward()
                    optimizer.step()

                    # --------------------- ------------------------------- --------
                    # ----------- calculate domain labels and losses ---------------
                    # --------------------- ------------------------------- --------
                    
                    # ------------------------------- domain part forward ------------------------
                    domain_outputs, domain_outputs_l1, domain_outputs_l2, \
                            domain_outputs_l3, w_main, w_l1, w_l2, w_l3, l1_rev, l2_rev, l3_rev\
                                            = model('dom_train', x1=domain_inputs, l=l, 
                                                     init_w_main=ini_w_main, init_w_l1=ini_w_l1, 
                                                     init_w_l2=ini_w_l2, init_w_l3=ini_w_l3)

                    domain_outputs_feat, domain_outputs_l1_feat, domain_outputs_l2_feat, \
                            domain_outputs_l3_feat, w_main_feat, w_l1_feat, w_l2_feat, w_l3_feat, \
                                    _, _, _ = model('dom_train', x1=domain_feat_inputs, l=l, 
                                                 init_w_main=ini_w_main, init_w_l1=ini_w_l1, 
                                                 init_w_l2=ini_w_l2, init_w_l3=ini_w_l3)

                    
                    domain_criterion = nn.BCEWithLogitsLoss()

                    domain_labels = domain_labels.squeeze()
                    domain_preds = torch.trunc(2*F.sigmoid(domain_outputs).data)

                    domain_preds_l1 = torch.trunc(2*F.sigmoid(domain_outputs_l1).data)
                    domain_preds_l2 = torch.trunc(2*F.sigmoid(domain_outputs_l2).data)
                    domain_preds_l3 = torch.trunc(2*F.sigmoid(domain_outputs_l3).data)
                    correct_domain = domain_labels.data

                    # ---------- Pytorch 0.2.0 edit change --------------------------
                    domain_counts += len(domain_preds)
                    domain_epoch_corrects += torch.sum(domain_preds == correct_domain)
                    domain_epoch_corrects_l1 += torch.sum(domain_preds_l1 == correct_domain)
                    domain_epoch_corrects_l2 += torch.sum(domain_preds_l2 == correct_domain)
                    domain_epoch_corrects_l3 += torch.sum(domain_preds_l3 == correct_domain)

                    domain_loss = domain_criterion(domain_outputs, domain_labels)
                    domain_loss_l1 = domain_criterion(domain_outputs_l1, domain_labels)
                    domain_loss_l2 = domain_criterion(domain_outputs_l2, domain_labels)
                    domain_loss_l3 = domain_criterion(domain_outputs_l3, domain_labels)

                    # ------ calculate pseudo predicts and losses with weights and threshold lambda -------

                    domain_epoch_loss += domain_loss.data[0]
                    domain_epoch_loss_l1 += domain_loss_l1.data[0]
                    domain_epoch_loss_l2 += domain_loss_l2.data[0]
                    domain_epoch_loss_l3 += domain_loss_l3.data[0]

                    w_main = w_main.expand_as(domain_loss)
                    w_l1 = w_l1.expand_as(domain_loss_l1)
                    w_l2 = w_l2.expand_as(domain_loss_l2)
                    w_l3 = w_l3.expand_as(domain_loss_l3)

                    # ------- domain classifier update ----------

                    dom_loss = torch.abs(w_main)*domain_loss+ \
                               torch.abs(w_l1)*domain_loss_l1 + \
                               torch.abs(w_l2)*domain_loss_l2+ \
                               torch.abs(w_l3)*domain_loss_l3

                    total_epoch_loss += dom_loss.data[0]
                    dom_optimizer.zero_grad()
                    dom_loss.backward(retain_graph=True)
                    dom_optimizer.step()

                    # ------- domain weights update ----------
                    if epoch >= pre_epochs:
                        dom_w_loss = w_main*domain_loss+ \
                                     w_l1*domain_loss_l1+ \
                                     w_l2*domain_loss_l2+ \
                                     w_l3*domain_loss_l3
                            
                        dom_w_optimizer.zero_grad()
                        dom_w_loss.backward()
                        dom_w_optimizer.step()

                    # --------------------- ------------------------------- --------
                    # ----------------- calculate domain feat losses ---------------
                    # --------------------- ------------------------------- --------


                    # ---------- domain feature update ----------

                    dom_feat_weight_tensor = dom_feat_weight.cuda()

                    domain_feat_criterion = nn.BCEWithLogitsLoss(weight=dom_feat_weight_tensor)

                    domain_preds_feat = torch.trunc(2*F.sigmoid(domain_outputs_feat).data)
                                        
                    domain_preds_l1_feat = torch.trunc(2*F.sigmoid(domain_outputs_l1_feat).data)
                    domain_preds_l2_feat = torch.trunc(2*F.sigmoid(domain_outputs_l2_feat).data)
                    domain_preds_l3_feat = torch.trunc(2*F.sigmoid(domain_outputs_l3_feat).data)
                    
                    domain_loss_feat = domain_feat_criterion(domain_outputs_feat, domain_labels)
                    domain_loss_l1_feat = domain_feat_criterion(domain_outputs_l1_feat, domain_labels)
                    domain_loss_l2_feat = domain_feat_criterion(domain_outputs_l2_feat, domain_labels)
                    domain_loss_l3_feat = domain_feat_criterion(domain_outputs_l3_feat, domain_labels)

                    domain_epoch_loss += domain_loss_feat.data[0]
                    domain_epoch_loss_l1 += domain_loss_l1_feat.data[0]
                    domain_epoch_loss_l2 += domain_loss_l2_feat.data[0]
                    domain_epoch_loss_l3 += domain_loss_l3_feat.data[0]

                    w_main_feat = w_main_feat.expand_as(domain_loss_feat)
                    w_l1_feat = w_l1_feat.expand_as(domain_loss_l1_feat)
                    w_l2_feat = w_l2_feat.expand_as(domain_loss_l2_feat)
                    w_l3_feat = w_l3_feat.expand_as(domain_loss_l3_feat)

                    dom_feat_loss = torch.abs(w_main_feat)*domain_loss_feat+ \
                                    torch.abs(w_l1_feat)*domain_loss_l1_feat + \
                                    torch.abs(w_l2_feat)*domain_loss_l2_feat+ \
                                    torch.abs(w_l3_feat)*domain_loss_l3_feat

                    total_epoch_loss += dom_feat_loss.data[0]
                    dom_feat_optimizer.zero_grad()
                    dom_feat_loss.backward()
                    dom_feat_optimizer.step()
                
            # ----------------------------------------------------------
            # ------------------ Testing Phase -------------------------
            # ----------------------------------------------------------

            elif phase == 'test' :
                model.train(False)  # Set model to evaluate mode
                model.eval()

                incorrect_count = 0
                for inputs, labels, path, t_dict in dset_loaders['test']:
                    
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    class_outputs = model('test', inputs)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # ------------ test classification statistics ------------
                    _, preds = torch.max(class_outputs.data, 1)
                    criterion = nn.CrossEntropyLoss()
                    class_loss = criterion(class_outputs, labels)
                    epoch_loss += class_loss.data[0]
                    epoch_corrects += torch.sum(preds == labels.data)

                    for i in range(args.num_class):
                        if labels.data[0] == i:
                            if i in test_corrects:
                                test_corrects[i] += torch.sum(preds == labels.data)
                            else:
                                test_corrects[i] = torch.sum(preds == labels.data)
                            if i in test_totals:
                                test_totals[i] += 1
                            else:
                                test_totals[i] = 1
            
            # ----------------  print statistics results   --------------------

            if phase == 'train':
                epoch_loss = epoch_loss / batch_count
                epoch_acc = epoch_corrects / class_count
                epoch_loss_s = epoch_loss
                epoch_acc_s = epoch_acc
                if epoch < pre_epochs:
                    if epoch == pre_epochs - 1:
                        pre_epoch_acc_s = (pre_epoch_acc_s + epoch_acc_s) / 2
                        pre_epoch_loss_s = (pre_epoch_loss_s + epoch_loss_s) / 2
                    else:
                        pre_epoch_acc_s = epoch_acc_s
                        pre_epoch_loss_s = epoch_loss_s

                else:
                    train_num = epoch - pre_epochs + 1
                    total_epoch_acc_s += epoch_acc_s
                    total_epoch_loss_s += epoch_loss_s
                    avg_epoch_acc_s = total_epoch_acc_s / train_num
                    avg_epoch_loss_s = total_epoch_loss_s / train_num

                domain_avg_loss = domain_epoch_loss / batch_count
                domain_avg_loss_l1 = domain_epoch_loss_l1 / batch_count
                domain_avg_loss_l2 = domain_epoch_loss_l2 / batch_count
                domain_avg_loss_l3 = domain_epoch_loss_l3 / batch_count

                domain_acc = domain_epoch_corrects / domain_counts
                domain_acc_l1 = domain_epoch_corrects_l1 / domain_counts
                domain_acc_l2 = domain_epoch_corrects_l2 / domain_counts
                domain_acc_l3 = domain_epoch_corrects_l3 / domain_counts

                total_avg_loss = total_epoch_loss / batch_count

                print('Phase: {} lr_mult: {:.4f} Loss: {:.4f} D_loss: {:.4f} D1_loss: {:.4f} D2_loss: {:.4f} D3_loss: {:.4f} Acc: {:.4f} D_Acc: {:.4f}'.format(
                      phase, epoch_lr_mult, epoch_loss, domain_avg_loss, 
                      domain_avg_loss_l1, domain_avg_loss_l2, domain_avg_loss_l3, 
                      epoch_acc, domain_acc))
                print("Total loss: {:.4f}, w_main: {:.4f}, 1: {:.4f}, 2: {:.4f}, 3: {:.4f}".format(
                      total_avg_loss, w_main.data[0],w_l1.data[0],w_l2.data[0],w_l3.data[0]))

                class_loss_point.append(float("%.4f" % epoch_loss))
                domain_loss_point.append(float("%.4f" % domain_avg_loss))
                source_acc_point.append(float("%.4f" % epoch_acc))
                domain_acc_point.append(float("%.4f" % domain_acc))
                lr_point.append(float("%.4f" % epoch_lr_mult))

                domain_loss_point_l1.append(float("%.4f" % domain_avg_loss_l1))
                domain_loss_point_l2.append(float("%.4f" % domain_avg_loss_l2))
                domain_loss_point_l3.append(float("%.4f" % domain_avg_loss_l3))

                domain_acc_point_l1.append(float("%.4f" % domain_acc_l1))
                domain_acc_point_l2.append(float("%.4f" % domain_acc_l2))
                domain_acc_point_l3.append(float("%.4f" % domain_acc_l3))

            else:
                epoch_loss = epoch_loss / len(dsets['test'])
                epoch_acc = epoch_corrects / len(dsets['test'])
                epoch_acc_t = epoch_acc
                print('Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                print(', '.join("%d:%d/%d"%(i,j,test_totals[i]) for i,j in test_corrects.items()))

                target_loss_point.append(float("%.4f" % epoch_loss))
                target_acc_point.append(float("%.4f" % epoch_acc))

            # deep copy the model, print best accuracy
            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                print('Best Test Accuracy: {:.4f}'.format(best_acc))

        print()

        # ------------------------------------ draw graph ------------------------------------
        # ------------------------------------------------------------------------------------

        try:
            os.makedirs('./graph/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        # ----------------------------------------------------------------

        # Create plots 2
        fig, ax = plt.subplots()
        ax.plot(epoch_point, source_acc_point, 'k', label='Source Classification Accuracy',color='r')
        ax.plot(epoch_point, domain_acc_point, 'k', label='Domain Accuracy',color='g')
        ax.plot(epoch_point, target_acc_point, 'k', label='Test Classification Accuracy',color='b')

        ax.annotate("ADV " + args.source_set + ' 2 ' + args.target_set + ' 0.5 Domain', xy=(1.05, 0.7), xycoords='axes fraction')
        ax.annotate('lr: %0.4f Pre epochs: %d Max epochs: %d' % (args.base_lr, pre_epochs, total_epochs), xy=(1.05, 0.65), xycoords='axes fraction')
        # ax.annotate('Pretrain epochs: %d' % PRETRAIN_EPOCH, xy=(1.05, 0.6), xycoords='axes fraction')
        # ax.annotate('Confidence Threshold: %0.3f' % confid_threshold, xy=(1.05, 0.55), xycoords='axes fraction')
        # ax.annotSate('Discriminator Threshold: %0.3f ~ %0.3f' % (LOW_DISCRIM_THRESH_T, UP_DISCRIM_THRESH_T), xy=(1.05, 0.5), xycoords='axes fraction')
        ax.annotate('L1,L2,L3,Main Disc_Weight: %0.4f %0.4f %0.4f %0.4f' % \
                    (w_l1.data[0], w_l2.data[0], w_l3.data[0], -1* w_main.data[0]), xy=(1.05, 0.5), xycoords='axes fraction')

        if epoch >= 49:
            ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
        if epoch >= 99:
            ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
            ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
        if epoch >= 199:
            ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
            ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
            ax.annotate('200 Epoch Accuracy: %0.4f' % (target_acc_point[199]), xy=(1.05, 0.25), xycoords='axes fraction')
        if epoch >= 299:
            ax.annotate('50 Epoch Accuracy: %0.4f' % (target_acc_point[49]), xy=(1.05, 0.35), xycoords='axes fraction')
            ax.annotate('100 Epoch Accuracy: %0.4f' % (target_acc_point[99]), xy=(1.05, 0.3), xycoords='axes fraction')
            ax.annotate('200 Epoch Accuracy: %0.4f' % (target_acc_point[199]), xy=(1.05, 0.25), xycoords='axes fraction')
            ax.annotate('300 Epoch Accuracy: %0.4f' % (target_acc_point[299]), xy=(1.05, 0.2), xycoords='axes fraction')
        if epoch >= total_epochs:
            ax.annotate('%d Epoch Accuracy: %0.4f' % (int(total_epochs),target_acc_point[total_epochs-1]), xy=(1.05, 0.15), xycoords='axes fraction')

        ax.annotate('Last Epoch Accuracy: %0.4f' % (epoch_acc), xy=(1.05, 0.1), xycoords='axes fraction', size=14)

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

        plt.savefig('graph/'+args.source_set+'2'+args.target_set+'_acc.png', bbox_inches='tight')
        
        if epoch % 50 == 0 or epoch == num_epochs -1:
            try:
                os.makedirs('./graph/'+args.source_set+'2'+args.target_set)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            plt.savefig('graph/'+args.source_set+'2'+args.target_set+'/'+'epoch'+str(epoch)+',acc'+str(epoch_acc_t)+'.png', bbox_inches='tight')

        fig.clf()

        plt.clf()

        epoch += 1
    time_elapsed = time.time() - since

    try:
        os.makedirs('./Result_txt/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return best_model

######################################################################
# Learning rate scheduler
def lr_scheduler(optimizer, lr_mult, weight_mult=1):
    counter = 0
    for param_group in optimizer.param_groups:
        if counter == 0:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult / 10.0
        else:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult
        counter += 1

    return optimizer, lr_mult

def dom_w_scheduler(optimizer, lr_mult, weight_mult=1):
    counter = 0
    for param_group in optimizer.param_groups:
        if counter == 0:
            optimizer.param_groups[counter]['lr'] = args.base_lr * lr_mult * weight_mult
        counter += 1

    return optimizer, lr_mult

def diff_states(dict_canonical, dict_subset):
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))
    #Sanity check that param names overlap
    #Note that params are not necessarily in the same order
    #for every pretrained model
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    assert len(not_in_1) == 0
    assert len(not_in_2) == 0

    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        assert hasattr(v2, 'size')
        if v1.size() != v2.size():
            yield (name, v1)

def load_model_merged(name, num_classes):

    model = models.__dict__[name](num_classes=num_classes)

    #Densenets don't (yet) pass on num_classes, hack it in for 169
    if name == 'densenet169':
        model = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, \
                                            block_config=(6, 12, 32, 32), num_classes=num_classes)

    if name == 'densenet201':
        model = torchvision.models.DenseNet(num_init_features=64, growth_rate=32, \
                                            block_config=(6, 12, 48, 32), num_classes=num_classes)
    if name == 'densenet161':
        model = torchvision.models.DenseNet(num_init_features=96, growth_rate=48, \
                                            block_config=(6, 12, 36, 24), num_classes=num_classes)

    pretrained_state = model_zoo.load_url(model_urls[name])

    #Diff
    diff = [s for s in diff_states(model.state_dict(), pretrained_state)]
    print("Replacing the following state from initialized", name, ":", \
          [d[0] for d in diff])

    for name, value in diff:
        pretrained_state[name] = value

    assert len([s for s in diff_states(model.state_dict(), pretrained_state)]) == 0

    #Merge
    model.load_state_dict(pretrained_state)
    return model, diff

def scale_gradients(v, weights): # assumes v is batch x ...
    def hook(g):
        return g*weights.view(*((-1,)+(len(g.size())-1)*(1,))) # probably nicer to hard-code -1,1,...,1
    v.register_hook(hook)

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

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def iterator_reset(iterator_name_i, dset_name_i, batchsize_i):
    if batchsize_i > 0:
        if args.useRatio:
            dset_loaders[iterator_name_i] = torch.utils.data.DataLoader(dsets[dset_name_i],
                                                batch_size=1, shuffle=True)
        else:
            dset_loaders[iterator_name_i] = torch.utils.data.DataLoader(dsets[dset_name_i],
                                                batch_size=batchsize_i, shuffle=True, drop_last=True)
        iterator = iter(dset_loaders[iterator_name_i])
    else:
        iterator = iter([])
    
    return iterator

def iterator_update(iterator_i, dset_name_i, batchsize_i, pointer_i, total_pointer_i,  i_type="pseu"):
    if args.useRatio:
        if pointer_i + batchsize_i > len(iterator_i):
            iterator_i = iterator_reset(iterator_i, dset_name_i, batchsize_i)
            pointer_i = 0

        iterator_batch_inputs = torch.FloatTensor([])
        iterator_batch_labels = torch.LongTensor([])
        iterator_batch_weights = torch.DoubleTensor([])
        iterator_batch_dom_confid = torch.DoubleTensor([])

        for i in range(batchsize_i):
            if i_type == "pseu":
                i_inputs, i_labels, _, i_dom_conf, i_weights, _  = iterator_i.next()
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
            iterator_i = iterator_reset(iterator_i, dset_name_i, batchsize_i)
            pointer_i = 0

        if i_type == "pseu":
            iterator_batch_inputs, iterator_batch_labels, _, i_dom_conf, iterator_batch_weights, _  = iterator_i.next()
            iterator_batch_dom_confid = 2*(1 - i_dom_conf)
        else:
            iterator_batch_inputs, iterator_batch_labels, _, _ = iterator_i.next()

        pointer_i += 1
        total_pointer_i += 1

    if i_type == "pseu":
        return iterator_i, iterator_batch_inputs, iterator_batch_labels, pointer_i, total_pointer_i, iterator_batch_weights, iterator_batch_dom_confid
    else:
        return iterator_i, iterator_batch_inputs, iterator_batch_labels, pointer_i, total_pointer_i
                   
def pull_back_to_one(x):
    if x > 1:
        x = 1
    return x

# model_names = model_urls.keys()

#------------------------ model 1 ------------------------------

model_pretrained, diff = load_model_merged('resnet50', args.num_class)

prev_w = args.form_w
last_w = args.main_w

model_dann = DISCRIMINATIVE_DANN(model_pretrained, prev_w, last_w, args.num_class)
# print(model_dann)

model_dann = model_dann.cuda()

for param in model_dann.parameters():
    param.requires_grad = True

# Observe that all parameters are being optimized
ignored_params_list = list(map(id, model_dann.source_bottleneck.parameters()))
ignored_params_list.extend(list(map(id, model_dann.source_classifier.parameters())))
ignored_params_list.extend(list(map(id, model_dann.domain_pred.parameters())))
ignored_params_list.extend(list(map(id, model_dann.disc_weight.parameters())))
ignored_params_list.extend(list(map(id, model_dann.disc_activate.parameters())))
ignored_params_list.extend(list(map(id, model_dann.domain_pred_l1.parameters())))
ignored_params_list.extend(list(map(id, model_dann.domain_pred_l2.parameters())))
ignored_params_list.extend(list(map(id, model_dann.domain_pred_l3.parameters())))
ignored_params_list.extend(list(map(id, model_dann.l1_bottleneck.parameters())))
ignored_params_list.extend(list(map(id, model_dann.l2_bottleneck.parameters())))
ignored_params_list.extend(list(map(id, model_dann.l3_bottleneck.parameters())))

base_params = filter(lambda p: id(p) not in ignored_params_list,
                     model_dann.parameters())

sub_dom_params_list = list(map(id, model_dann.domain_pred_l1.parameters()))
sub_dom_params_list.extend(list(map(id, model_dann.domain_pred_l2.parameters())))
sub_dom_params_list.extend(list(map(id, model_dann.domain_pred_l3.parameters())))
sub_dom_params_list.extend(list(map(id, model_dann.l1_bottleneck.parameters())))
sub_dom_params_list.extend(list(map(id, model_dann.l2_bottleneck.parameters())))
sub_dom_params_list.extend(list(map(id, model_dann.l3_bottleneck.parameters())))

sub_dom_params = filter(lambda p: id(p) in sub_dom_params_list,
                     model_dann.parameters())
# print(list(model_dann.parameters()))

optimizer_cls = optim.SGD([
            {'params': base_params},
            {'params': model_dann.source_bottleneck.parameters(), 'lr': args.base_lr},
            {'params': model_dann.source_classifier.parameters(), 'lr': args.base_lr},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay, nesterov= args.nesterov)

optimizer_dom = optim.SGD([
            {'params': sub_dom_params},
            {'params': model_dann.source_bottleneck.parameters(), 'lr': args.base_lr},
            {'params': model_dann.domain_pred.parameters(), 'lr': args.base_lr},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay, nesterov= args.nesterov)

optimizer_dom_feature = optim.SGD([
            {'params': base_params},
            ], lr=args.base_lr / 10.0, momentum=0.9, weight_decay=args.decay, nesterov= args.nesterov)

optimizer_dom_w = optim.SGD(model_dann.disc_weight.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0, nesterov= args.nesterov)

optimizer_pseudo = optim.SGD(model_dann.disc_activate.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=3e-4)

#------------------------ train models ------------------------------

model_dann = train_model(model_dann, optimizer_cls, optimizer_dom, optimizer_dom_w, optimizer_dom_feature, 
                         lr_scheduler, dom_w_scheduler, base_params, num_epochs=total_epochs)
