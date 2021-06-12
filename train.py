"""
Parts of this script has been copied from https://github.com/VisionLearningGroup/DANCE
"""
from __future__ import print_function
import yaml
import easydict
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data_loader.get_loader import get_loader_with_noise
from utils.utils import *
from utils.lr_schedule import inv_lr_scheduler
from utils.loss import *
import logging
import random

import numpy as np
import warnings

warnings.simplefilter('ignore')

# Training settings

import argparse

parser = argparse.ArgumentParser(description='Pytorch DA',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

parser.add_argument('--source_path', type=str, default='./utils/source_list.txt', metavar='B',
                    help='path to source list')
parser.add_argument('--target_path', type=str, default='./utils/target_list.txt', metavar='B',
                    help='path to target list')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--exp_name', type=str, default='office_close', help='/path/to/config/file')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

parser.add_argument('--noise-type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--percent', type=float, default=0.2,
                    help='Percentage of noise')
# args = parser.parse_args()
args = parser.parse_args()
config_file = args.config
conf = yaml.load(open(config_file))
save_config = yaml.load(open(config_file))
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
args.cuda = torch.cuda.is_available()
source_data = args.source_path
target_data = args.target_path
evaluation_data = args.target_path

batch_size = conf.data.dataloader.batch_size
filename = source_data.split("_")[1] + "2" + target_data.split("_")[1]
filename = os.path.join("result", f"{args.noise_type}_{args.percent}",
                        config_file.replace(".yaml", ""), filename)
if not os.path.exists(filename):
    os.makedirs(filename)
    os.makedirs(filename+'/plots')
print("record in %s " % filename)

data_transforms = {
    source_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    target_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    evaluation_data: transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = torch.cuda.is_available()
source_loader, target_loader, \
test_loader, target_folder = get_loader_with_noise(source_data, target_data, 
                                        evaluation_data, data_transforms, 
                                        noisify, conf.data.dataset.n_share+conf.data.dataset.n_source_private, 
                                        noise_type=args.noise_type, noise_rate=args.percent, 
                                        batch_size=batch_size, return_id=True,
                                        balanced=conf.data.dataloader.class_balance)
dataset_test = test_loader
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
num_class = n_share + n_source_private

G, C1, C2 = get_model_mme_2head(conf.model.base_model, num_class=num_class,
                      temp=conf.model.temp)
device = torch.device("cuda")
if args.cuda:
    G.cuda()
    C1.cuda()
    C2.cuda()

G.to(device)
C1.to(device)
C2.to(device)
ndata = target_folder.__len__()

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad and "features" in key:
        if 'bias' in key:
            params += [{'params': [value], 'lr': conf.train.multi,
                        'weight_decay': conf.train.weight_decay}]
        else:
            params += [{'params': [value], 'lr': conf.train.multi,
                        'weight_decay': conf.train.weight_decay}]
    else:
        if 'bias' in key:
            params += [{'params': [value], 'lr': 1.0,
                        'weight_decay': conf.train.weight_decay}]
        else:
            params += [{'params': [value], 'lr': 1.0,
                        'weight_decay': conf.train.weight_decay}]

criterion = torch.nn.CrossEntropyLoss().cuda()

opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                  weight_decay=0.0005, nesterov=True)
opt_c1 = optim.SGD(list(C1.parameters()), lr=1.0,
                   momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                   nesterov=True)
opt_c2 = optim.SGD(list(C2.parameters()), lr=1.0,
                   momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                   nesterov=True)

G = nn.DataParallel(G)
C1 = nn.DataParallel(C1)
C2 = nn.DataParallel(C2)

param_lr_g = []
for param_group in opt_g.param_groups:
    param_lr_g.append(param_group["lr"])
    
param_lr_f1 = []
for param_group in opt_c1.param_groups:
    param_lr_f1.append(param_group["lr"])

param_lr_f2 = []
for param_group in opt_c2.param_groups:
    param_lr_f2.append(param_group["lr"])

def train():
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_f1, opt_c1, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_f2, opt_c2, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        label_t = data_t[1]
        index_t = data_t[2]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        index_t = Variable(index_t.cuda())
        index_t = Variable(index_t.cuda())
        if len(img_t) < batch_size:
            break
        if len(img_s) < batch_size:
            break
        opt_g.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()
        ## Weight normalizztion
        C1.module.weight_norm()
        C2.module.weight_norm()


        ## Step A-1
        feat = G(img_s)
        out_s1 = C1(feat)
        out_s2 = C2(feat)

        if step > 100:
            loss_s = loss_select(out_s1, out_s2, label_s, args.percent)
        else:
            loss_s = criterion(out_s1, label_s) + criterion(out_s2, label_s)

        ## Step A-2
        feat_t = G(img_t)
        out_t1 = C1(feat_t)
        out_t2 = C2(feat_t)

        loss_crsent, mask_known = cross_entropy_margin(out_t1, out_t2, 2*conf.train.thr, 2*conf.train.margin)

        loss_ent = entropy_margin(out_t1, out_t2, conf.train.thr, conf.train.margin)

        if step > 100:
            all = loss_s + conf.train.eta * loss_crsent  + conf.train.eta * loss_ent
        else:
            all = loss_s + conf.train.eta * loss_ent

        opt_g.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        all.backward()
        opt_g.step()
        opt_c1.step()
        opt_c2.step()
        
        if step > 100:
            ## Step B
            feat_t = G(img_t)
            out_t1 = C1(feat_t)
            out_t2 = C2(feat_t)

            prob_t1 = torch.softmax(out_t1, dim=1)
            prob_t2 = torch.softmax(out_t2, dim=1)

            loss_crsent = torch.mean(torch.clamp(cross_entropy(prob_t1, prob_t2, reduce=False) + cross_entropy(prob_t2, prob_t1, reduce=False) , max=4*conf.train.thr))

            feat = G(img_s)
            out_s1 = C1(feat)
            out_s2 = C2(feat)

            loss_s = loss_select(out_s1, out_s2, label_s, args.percent)

            all = loss_s - conf.train.eta * loss_crsent

            opt_c1.zero_grad()
            opt_c2.zero_grad()

            all.backward()
            opt_c1.step()
            opt_c2.step()

            ## Step C
            for j in range(4):
                feat_t = G(img_t)
                out_t1 = C1(feat_t)
                out_t2 = C2(feat_t)

                prob_t1 = torch.softmax(out_t1, dim=1)
                prob_t2 = torch.softmax(out_t2, dim=1)

                loss_crsent = torch.mean(torch.clamp(cross_entropy(prob_t1, prob_t2, reduce=False) + cross_entropy(prob_t2, prob_t1, reduce=False) , max=4*conf.train.thr)[mask_known])

                all = conf.train.eta * loss_crsent
                opt_g.zero_grad()

                all.backward()
                opt_g.step()
        
        if step % conf.train.log_interval == 0:
            print('Train [{}/{} ({:.2f}%)]\tLoss Source: {:.6f}\tLoss Crs: {:.6f}\t'.format(
                step, conf.train.min_step,
                100 * float(step / conf.train.min_step),
                loss_s.item(), loss_crsent.item()))

        if step > 0 and step % 100 == 0:
            test(step, dataset_test, filename, n_share, num_class, G, C1, C2,
                 conf.train.thr)
            G.train()
            C1.train()
            C2.train()

def test(step, dataset_test, filename, n_share, unk_class, G, C1, C2, threshold):
    G.eval()
    C1.eval()
    C2.eval()

    class_list = [i for i in range(n_share)]
    class_list.append(unk_class)

    all_prob1 = []
    all_prob2 = []
    all_gt = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            out_t1 = C1(feat)
            out_t1 = F.softmax(out_t1)
            entr1 = -torch.sum(out_t1 * torch.log(out_t1), 1).data.cpu().numpy()
            out_t2 = C2(feat)
            out_t2 = F.softmax(out_t2)

            all_gt += list(label_t.data.cpu().numpy())

            for i in range(len(out_t1)):
                all_prob1.append(out_t1[i].cpu().numpy())
                all_prob2.append(out_t2[i].cpu().numpy())

    targets = np.array(all_gt)
    probs = np.array(all_prob1)
    probs2 = np.array(all_prob2)

    pred = np.argmax(probs+probs2, axis=1)

    max_acc = 0
    pred_best = pred

    ent = cross_entropy(torch.tensor(probs).float(), torch.tensor(probs2).float(), reduce=False) + cross_entropy(torch.tensor(probs2).float(), torch.tensor(probs).float(), reduce=False)
    ent = ent.numpy()
    
    th = threshold * 2
    pred_ = np.argmax(probs, axis=1)

    pred_[ent>th] = unk_class

    cm = extended_confusion_matrix(targets, pred_, true_labels=range(unk_class+1), pred_labels=range(unk_class+1))

    acc_all = (np.sum([cm[i][i] for i in range(unk_class)]) + cm[unk_class][unk_class]) / np.sum(cm)
    acc_unk = cm[unk_class][unk_class] / np.sum(cm[unk_class,:])

    cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = np.mean([cm[i][i] for i in range(unk_class) if not np.isnan(cm[i][i])])
    acc_os = np.mean([cm[i][i] for i in range(unk_class+1) if not np.isnan(cm[i][i])])

    cm = extended_confusion_matrix(targets[targets<unk_class], np.argmax(probs, axis=1)[targets<unk_class], true_labels=range(unk_class), pred_labels=range(unk_class))
    cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
    acc_known = np.mean([cm[i][i] for i in range(unk_class) if not np.isnan(cm[i][i])])

    print(f'Step: {step} #Test data {len(targets)}, Test acc {acc_os:.4f}, Test acc star {acc_os_star:.4f} Unk Acc. {acc_unk:.4f} Kwn Acc. {acc_known:.4f}')

    output = [step, acc_os, acc_os_star, acc_unk, acc_all]
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename+'/log', format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info(output)

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def loss_select(y_1, y_2, t, forget_rate, co_lambda=0.1):
    loss_pick_1 = F.cross_entropy(y_1, t, reduce = False) 
    loss_pick_2 = F.cross_entropy(y_2, t, reduce = False)
    loss_pick = loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2, reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)


    ind_sorted = torch.argsort(loss_pick)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss

train()
