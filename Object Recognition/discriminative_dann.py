import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import math
import copy
from collections import OrderedDict
from torch.autograd import Variable, Function
import numpy as np

BATCH_SIZE = 16
INI_DISC_SIG_SCALE = 0.1
INI_DISC_A = 1
LAST_WEIGHT_LIMIT = -2
INTEGRAL_SIGMA_VAR = 0.1


class Calc_Prob(nn.Module):

    def __init__(self, initWeightScale, initBias):

        super(Calc_Prob, self).__init__()

        self.dom_func_sigma = nn.Parameter(torch.ones(1),requires_grad=False)
        self.integral_var = Variable(torch.FloatTensor([INTEGRAL_SIGMA_VAR]).cuda())
        # nn.Parameter(torch.zeros(1),requires_grad=True)
 
        self.sigma_scale = initWeightScale
    def forward(self, class_t, dom_res):

        # sig = (self.dom_func_sigma * self.sigma_scale).expand_as(dom_res)
        # def_sig = self.integral_var.expand_as(dom_res)

        # dom_prob = F.sigmoid(dom_res.squeeze())
        # exp_x = -(dom_prob - 0.5)**2 / sig**2

        # integralLimit = (math.pi**0.5 * torch.erf(0.5/def_sig)) / (1/def_sig)

        # dom_integral = (math.pi**0.5 * torch.erf(0.5/sig)) / (1/sig)

        # a = integralLimit / dom_integral

        # dom_weight = a * torch.exp(exp_x)

        top_prob, top_label = torch.topk(F.softmax(class_t.squeeze()), 1)

        confid_rate = top_prob.squeeze()
        
        # act_weight = confid_rate * dom_weight

        # final_weight = act_weight

        return confid_rate
        # final_weight, dom_weight,  sig, a


class Discriminator_Weights_Adjust(nn.Module):

    def __init__(self, form_weight, last_weight):

        super(Discriminator_Weights_Adjust, self).__init__()

        self.main_var = Variable(torch.FloatTensor([0]).cuda())
        self.l1_var = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.l2_var = nn.Parameter(torch.zeros(1),requires_grad=True)
        default_const = last_weight - form_weight

        self.k_var = Variable(torch.FloatTensor([default_const]).cuda())

        self.f_weight = form_weight
        self.l_weight = last_weight

    def forward(self, main_weight, l1_weight, l2_weight, l3_weight):

        w_main = main_weight + self.main_var

        w_l1 = l1_weight + self.l1_var
        w_l2 = l2_weight + self.l2_var

        if abs(w_l1.data[0]) > self.f_weight:
            w_l1 = w_l1 - np.sign(w_l1.data[0]) * (abs(w_l1.data[0]) - self.f_weight)        
        if abs(w_l2.data[0]) > self.f_weight:
            w_l2 = w_l2 - np.sign(w_l2.data[0]) * (abs(w_l2.data[0]) - self.f_weight)  

        w_l3 = (w_main - self.k_var) - w_l1 - w_l2

        l1_rev = np.sign(w_l1.data[0])
        l2_rev = np.sign(w_l2.data[0])
        l3_rev = np.sign(w_l3.data[0])

        # if (w_l3.data[0] < LAST_WEIGHT_LIMIT):
        #     total_exceed = w_l3 - LAST_WEIGHT_LIMIT
        #     w_l1_ratio = w_l1 / (w_l1 + w_l2) 
        #     w_l2_ratio = w_l2 / (w_l1 + w_l2)
        #     self.l1_var += total_exceed * w_l1_ratio
        #     self.l2_var += total_exceed * w_l2_ratio
        #     w_l3 = (w_main - self.k_var) - w_l1 - w_l2

        return w_main, w_l1, w_l2, w_l3, l1_rev, l2_rev, l3_rev

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

class DISCRIMINATIVE_DANN(nn.Module):
    def __init__(self, pre_trained, form_w, last_w, classes):
        super(DISCRIMINATIVE_DANN, self).__init__()
        self.num_class = classes
        self.form_weight = form_w
        self.last_weight = last_w

        self.disc_activate = Calc_Prob(INI_DISC_SIG_SCALE, INI_DISC_A)
        self.disc_weight = Discriminator_Weights_Adjust(self.form_weight, self.last_weight)

        self.conv1 = pre_trained.conv1
        self.bn1 = pre_trained.bn1
        self.relu = pre_trained.relu
        self.maxpool = pre_trained.maxpool

        self.layer1 = pre_trained.layer1
        self.layer2 = pre_trained.layer2
        self.layer3 = pre_trained.layer3
        self.layer4 = pre_trained.layer4

        self.domain_pred = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))
        self.domain_pred_l1 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))
        self.domain_pred_l2 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))
        self.domain_pred_l3 = nn.Sequential(nn.Linear(256, 3072), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(3072, 2048), nn.ReLU(True), nn.Dropout(),
                                         nn.Linear(2048, 1))

        self.process = pre_trained.avgpool

        self.process_l1 = nn.AvgPool2d(kernel_size=56)
        self.process_l2 = nn.AvgPool2d(kernel_size=28)
        self.process_l3 = nn.AvgPool2d(kernel_size=14)

        self.source_bottleneck = nn.Sequential(nn.Linear(pre_trained.fc.in_features, 256))

        self.l1_bottleneck = nn.Sequential(nn.Linear(256, 256))
        self.l2_bottleneck = nn.Sequential(nn.Linear(512, 256))
        self.l3_bottleneck = nn.Sequential(nn.Linear(1024, 256))

        self.source_classifier = nn.Sequential(nn.Linear(256, self.num_class))

        # ----- data parallel (multi-gpu) -------
        # self.conv1 = nn.DataParallel(self.conv1)
        # self.layer1 = nn.DataParallel(self.layer1)
        # self.layer2 = nn.DataParallel(self.layer2)
        # self.layer3 = nn.DataParallel(self.layer3)
        # self.layer4 = nn.DataParallel(self.layer4)

        # self.source_bottleneck = nn.DataParallel(self.source_bottleneck)
        # self.l1_bottleneck = nn.DataParallel(self.l1_bottleneck)
        # self.l2_bottleneck = nn.DataParallel(self.l2_bottleneck)
        # self.l3_bottleneck = nn.DataParallel(self.l3_bottleneck)

        # self.domain_pred = nn.DataParallel(self.domain_pred)
        # self.domain_pred_l1 = nn.DataParallel(self.domain_pred_l1)
        # self.domain_pred_l2 = nn.DataParallel(self.domain_pred_l2)
        # self.domain_pred_l3 = nn.DataParallel(self.domain_pred_l3)

        # self._initialize_weights()

    def forward(self, cond, x1, x2=None, l=None,
                init_weight=None, init_w_main=None, init_w_l1=None,
                init_w_l2=None, init_w_l3=None,):

        base1 = self.conv1(x1)
        base1 = self.bn1(base1)
        base1 = self.relu(base1)
        base1 = self.maxpool(base1)
        l1 = self.layer1(base1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        process = self.process(l4).view(l4.size(0), -1)

        if (cond == 'pseudo_discriminator'):
            
            bottle = self.source_bottleneck(process)
            class_pred = self.source_classifier(bottle)
            dom_pred = self.domain_pred(bottle)

            conf = self.disc_activate(class_pred, dom_pred)

            return class_pred, dom_pred.squeeze(), conf

        elif (cond == 'pretrain'):

            bottle = self.source_bottleneck(process)
            class_pred = self.source_classifier(bottle)

            base2 = self.conv1(x2)
            base2 = self.bn1(base2)
            base2 = self.relu(base2)
            base2 = self.maxpool(base2)
            l1_2 = self.layer1(base2)
            l2_2 = self.layer2(l1_2)
            l3_2 = self.layer3(l2_2)
            l4_2 = self.layer4(l3_2)

            process_2 = self.process(l4_2)
            process_2 = process_2.view(l4_2.size(0), -1)
            bottle_2 = self.source_bottleneck(process_2)
            grad_inverse_hook_2 = bottle_2.register_hook(lambda grad: grad * -1*l)

            process_l1 = self.process_l1(l1_2).view(l1_2.size(0), -1)
            bottle_l1 = self.l1_bottleneck(process_l1)

            process_l2 = self.process_l2(l2_2).view(l2_2.size(0), -1)
            bottle_l2 = self.l2_bottleneck(process_l2)

            process_l3 = self.process_l3(l3_2).view(l3_2.size(0), -1)
            bottle_l3 = self.l3_bottleneck(process_l3)

            disc_main, disc_l1, disc_l2, disc_l3, l1_rev, l2_rev, l3_rev = self.disc_weight(init_w_main, init_w_l1,
                                                                    init_w_l2, init_w_l3)


            bottle_reverse = grad_reverse(bottle, l*-1)
            bottle_l1 = grad_reverse(bottle_l1, l*l1_rev)
            bottle_l2 = grad_reverse(bottle_l2, l*l2_rev)
            bottle_l3 = grad_reverse(bottle_l3, l*l3_rev)

            dom_pred = self.domain_pred(bottle_2)
            dom_pred_l1 = self.domain_pred_l1(bottle_l1)
            dom_pred_l2 = self.domain_pred_l2(bottle_l2)
            dom_pred_l3 = self.domain_pred_l3(bottle_l3)

            return class_pred, dom_pred.squeeze(), dom_pred_l1.squeeze(), \
                               dom_pred_l2.squeeze(), dom_pred_l3.squeeze(), \
                               disc_main, disc_l1, disc_l2, disc_l3, l1_rev, l2_rev, l3_rev
        
        elif (cond == 'cls_train'):
            
            bottle = self.source_bottleneck(process)
            class_pred = self.source_classifier(bottle)

            return class_pred

        elif (cond == 'dom_train'):
            bottle = self.source_bottleneck(process)

            process_l1 = self.process_l1(l1).view(l1.size(0), -1)
            bottle_l1 = self.l1_bottleneck(process_l1)

            process_l2 = self.process_l2(l2).view(l2.size(0), -1)
            bottle_l2 = self.l2_bottleneck(process_l2)

            process_l3 = self.process_l3(l3).view(l3.size(0), -1)
            bottle_l3 = self.l3_bottleneck(process_l3)

            disc_main, disc_l1, disc_l2, disc_l3, l1_rev, l2_rev, l3_rev = self.disc_weight(init_w_main, init_w_l1,
                                                                    init_w_l2, init_w_l3)
            
            bottle_reverse = grad_reverse(bottle, l*-1)
            bottle_l1 = grad_reverse(bottle_l1, l*l1_rev)
            bottle_l2 = grad_reverse(bottle_l2, l*l2_rev)
            bottle_l3 = grad_reverse(bottle_l3, l*l3_rev)

            dom_pred = self.domain_pred(bottle_reverse)
            dom_pred_l1 = self.domain_pred_l1(bottle_l1)
            dom_pred_l2 = self.domain_pred_l2(bottle_l2)
            dom_pred_l3 = self.domain_pred_l3(bottle_l3)

            return dom_pred.squeeze(), dom_pred_l1.squeeze(), \
                   dom_pred_l2.squeeze(), dom_pred_l3.squeeze(), \
                   disc_main, disc_l1, disc_l2, disc_l3, l1_rev, l2_rev, l3_rev

        else:
            # test and pseudo label
            bottle = self.source_bottleneck(process)
            class_pred = self.source_classifier(bottle)

            return class_pred
