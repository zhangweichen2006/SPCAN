import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics','ucf101-10','hmdb51-10'])

parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('modality2', type=str, choices=['RGB', 'Flow', 'RGBDiff'])

parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)

parser.add_argument('train_list2', type=str)
parser.add_argument('val_list2', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--num_segments', type=int, default=3)

# faster 25
parser.add_argument('--test_segments', type=int, default=15)

parser.add_argument('--test_crops', type=int, default=1)

parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')

parser.add_argument('--dropout2', '--do2', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')

parser.add_argument('--test_dropout', type=float, default=0.7)


parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])

parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--epochs2', default=45, type=int, metavar='N2',
                    help='number of total epochs2 to run')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--lr2', '--learning-rate2', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--save_freq', '-sf', default=10, type=int,
                    metavar='N', help='evaluation frequency (default: 10)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)

parser.add_argument('--form_w', type=float, default=0.4)
parser.add_argument('--main_w', type=float, default=-0.5)

parser.add_argument('--wp', type=float, default=0.055)
parser.add_argument('--wt', type=float, default=1)

parser.add_argument('--select', type=str, default='1-2')

parser.add_argument('--usePreT2D', type=str2bool, default=False)

parser.add_argument('--useT1DorT2', type=str, default="T2")

parser.add_argument('--diffS', type=str2bool, default=False)

parser.add_argument('--diffDFT2', type=str2bool, default=False)

parser.add_argument('--useT2CompD', type=str2bool, default=False)
parser.add_argument('--usemin', type=str2bool, default=False)

parser.add_argument('--useRatio', type=str2bool, default=False)

parser.add_argument('--useCurrentIter', type=str2bool, default=False)
# parser.add_argument('--useEpoch', type=bool, default=False)

parser.add_argument('--useLargeLREpoch', type=str2bool, default=True)

parser.add_argument('--MaxStep', type=int, default=0)

parser.add_argument('--useSepTrain', type=str2bool, default=True)

parser.add_argument('--fixW', type=str2bool, default=False)
parser.add_argument('--decay', type=float, default=0.0003)
parser.add_argument('--nesterov', type=str2bool, default=True)

parser.add_argument('--ReTestSource', type=str2bool, default=False)

parser.add_argument('--sourceTestIter', type=int, default=2000)
parser.add_argument('--defaultPseudoRatio', type=float, default=0.2)

parser.add_argument('--defaultPseudoRatio2', type=float, default=0.2)

parser.add_argument('--totalPseudoChange', type=int, default=100)


parser.add_argument('--evalBreakIter', type=int, default=200)

parser.add_argument('--step', type=float, default=0.0003)

parser.add_argument('--reverse_epoch_ratio', type=float, default=1)

parser.add_argument('--lr_ratio', type=float, default=0.5)

parser.add_argument('--dom_weight', type=float, default=0.1)

parser.add_argument('--pre_ratio', type=float, default=0.2)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--usingDoubleDecrease', type=str2bool, default=False)

parser.add_argument('--usingDoubleDecrease2', type=str2bool, default=False)

parser.add_argument('--pseudo_ratio', type=float, default=1.0)
parser.add_argument('--pseudo_ratio2', type=float, default=1.0)

parser.add_argument('--max_pseudo', type=float, default=0.5)
parser.add_argument('--max_pseudo2', type=float, default=0.5)

parser.add_argument('--usePrevAcc', type=str2bool, default=False)

parser.add_argument('--skip', type=int, default=1)

parser.add_argument('--useT1Only', type=str2bool, default=False)

parser.add_argument('--usingTriDec', type=str2bool, default=False)

parser.add_argument('--usingTriDec2', type=str2bool, default=False)