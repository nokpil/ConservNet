# coding=utf-8
import argparse
import os

import random



import sys
import numpy as np
import scipy as sc




import torch
import torch.nn as nn
import torch.nn.parallel

import torch.distributed as dist
import torch.optim

import torch.utils.data


from torch.utils.data import (
    DataLoader,
)  # (testset, batch_size=4,shuffle=False, num_workers=4)

from torch.utils.data.dataset import TensorDataset

import pickle

import tracemalloc
import distutils
import distutils.util

import sys
import src.DataStructure as DS
import src.MachineLearning as ML
from src.utils import *
from src.system import *
from src.MachineLearning import DCN, CN, TTC, ns, plot_now, imshow_now, scatter_now

def str2bool(v):
    return bool(distutils.util.strtobool(v))

parser = argparse.ArgumentParser(description="Pytorch ConservNet Training")

parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=10000, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)

parser.add_argument(
    "--lr",
    "--learning-rate",
    default=5e-5,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.01)",
    dest="weight_decay",
)
parser.add_argument(
    "--model", "--model", default="Siam", type=str, help="simulation data type : Con(servNet), Siam"
)
parser.add_argument(
    "--system", default="S1", type=str, help="simulation sytem, S1, S2, S3"
)

parser.add_argument("--iter", default=10, type=int, help="iter num")

parser.add_argument("--n", default=10, type=int, help="group num")
parser.add_argument("--m", default=200, type=int, help="data num")

parser.add_argument("--noise", default=0., type=float, help="noise strength")

parser.add_argument(
    "--indicator", default="", type=str, help="Additional specification for file name."
)
parser.add_argument("--seed", default=0, type=int, help="Random seed for torch and numpy")

class SiameseNet(nn.Module):
    def __init__(self, cfg_clf, block_type, D_agent):
        super(SiameseNet, self).__init__()
        self.classifier = cfg_Block(block_type, cfg_clf, D_agent, 'RL', False, False)
        self.final = nn.Linear(1, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data.fill_(0.1)
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.classifier(x)
        return out
    
    def predict(self, x):
        out = self.final(x)
        return out

# Standard
# Train / Test set shares CONSTANT 

# Standard
# Train / Test set shares CONSTANT 

def _gs(batch_size, batch_num):
    total_size = int(batch_size * batch_num)
    order = np.zeros(total_size)
    fixed = set()
    if batch_num > 1:
        for i in range(batch_num - 1):
            vacant = set.difference(set(np.arange(total_size)), set(np.arange(i*batch_size, (i+1)*batch_size)))
            seat = set.difference(vacant, fixed)
            selected = np.random.permutation(list(seat))[:batch_size]
            order[i * batch_size: (i + 1) * batch_size] = selected
            fixed = set.union(fixed, selected)
        # for final one
        i = batch_num - 1
        vacant = set.difference(set(np.arange(total_size)), set(np.arange(i * batch_size, (i + 1) * batch_size)))
        seat = set.difference(vacant, fixed)
        resolve = set.difference(set(np.arange(i * batch_size, (i + 1) * batch_size)), fixed)
        for p in list(resolve):
            for q, x in enumerate(order):
                if x < i * batch_size:
                    order[q] = p
                    fixed = set.difference(fixed, set([x]))
                    break

        seat = set.difference(vacant, fixed)
        assert len(seat) == batch_size
        selected = np.random.permutation(list(seat))[:batch_size]
        order[i * batch_size:(i+1) * batch_size] = selected
    return order

def group_shuffle(batch_size, batch_num, pos=False, single=False):
    if single:
        if pos: 
            return list(np.arange(batch_num))
        else:
            return list(np.random.permutation(np.arange(batch_num)))
            
    else:            
        if pos:
            order = []
            for i in range(batch_num):
                tmp = np.array(_gs(1, batch_size)) + int(i * batch_size)
                order = order + list(tmp)
            return np.array(order).astype('int')
        else:
            return _gs(batch_size, batch_num).astype('int')
        

class DataGen():
    def __init__(self, system_type, batch_size, batch_num):
        self.system = system_type(batch_size, batch_num)
    
    def run(self, file_name, total_size, batch_size, train_ratio, noise_strength=0):

        train_image1 = []
        train_answer1 = []
        train_label = []
        
        test_image1 = []
        test_answer1 = []
        test_label = []
        
        batch_num = int(total_size / batch_size)
        train_batch_size = int(batch_size * train_ratio)
        test_batch_size = int(batch_size * (1 - train_ratio))
        # print(train_batch_size, test_batch_size)
        for i in range(batch_num):
            for j in range(batch_size):
                data, answer = next(self.system)
                if j < batch_size * train_ratio:
                    train_image1.append(data.astype(float))
                    train_answer1.append(answer)
                else:
                    test_image1.append(data.astype(float))
                    test_answer1.append(answer)
        
        print('generating finished')
        # shuffle
        train_image1 = np.array(train_image1)
        train_answer1 = np.array(train_answer1)
        test_image1 = np.array(test_image1)
        test_answer1 = np.array(test_answer1)
        
        if train_batch_size == 1:
            single = True
            print('single')
        else:
            single = False
        
        pos_order = group_shuffle(train_batch_size, batch_num, pos=True, single=single)
        neg_order = group_shuffle(train_batch_size, batch_num, pos=False, single=single)
        train_image2_pos = train_image1[pos_order]
        train_image2_neg = train_image1[neg_order]
        train_answer2_pos = train_answer1[pos_order]
        train_answer2_neg = train_answer1[neg_order]
        train_label = list(np.zeros(len(train_image1))) + list(np.ones(len(train_image1)))
        train_image2 = list(train_image2_pos) + list(train_image2_neg)
        train_answer2 = list(train_answer2_pos) + list(train_answer2_neg)
        train_image1 = list(train_image1) + list(train_image1)
        train_answer1 = list(train_answer1) + list(train_answer1)

        pos_order = group_shuffle(test_batch_size, batch_num, pos=True, single=single)
        neg_order = group_shuffle(test_batch_size, batch_num, pos=False, single=single)
        test_image2_pos = test_image1[pos_order]
        test_image2_neg = test_image1[neg_order]
        test_answer2_pos = test_answer1[pos_order]
        test_answer2_neg = test_answer1[neg_order]
        test_label = list(np.zeros(len(test_image1))) + list(np.ones(len(test_image1)))
        test_image2 = list(test_image2_pos) + list(test_image2_neg)
        test_answer2 = list(test_answer2_pos) + list(test_answer2_neg)
        test_image1 = list(test_image1) + list(test_image1)
        test_answer1 = list(test_answer1) + list(test_answer1)
        
        train_output = {'Image1': train_image1, 'Image2': train_image2, 'Answer1': train_answer1, 'Answer2': train_answer2, 'Label': train_label}
        test_output = {'Image1': test_image1, 'Image2': test_image2, 'Answer1': test_answer1, 'Answer2': test_answer2, 'Label': test_label}
                
        # Output pickle
        with open('./data/' + file_name + '_train.pkl', 'wb') as f:
            pickle.dump(train_output, f)

        with open('./data/' + file_name + '_test.pkl', 'wb') as f:
            pickle.dump(test_output, f)


def train(model, train_loader, criterion, optimizer):
    train_losses = AverageMeter("TrainLoss", ":.4e")
    train_acc = AverageMeter("TrainAcc", ":.4e")
    for image1, image2, answer1, answer2, label in train_loader:
        image1 = image1.cuda()
        image2 = image2.cuda()
        label = label.cuda()
        d1 = model(image1)
        d2 = model(image2)
        pred = model.predict((d1 - d2)**2).squeeze(-1)
        train_loss = criterion(pred, label)
        cls = torch.where(torch.sigmoid(pred) > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses.update(train_loss.item(), image1.shape[0])
        train_acc.update(torch.mean((cls == label).float()) * 100, image1.shape[0])
    return train_losses.avg, train_acc.avg

def test(model, test_loader, criterion):
    test_losses = AverageMeter("TestLoss", ":.4e")
    test_acc = AverageMeter("TestAccuracy", ":.4e")
    for image1, image2, answer1, answer2, label in test_loader:
        label = label.cuda()
        image1 = image1.cuda()
        image2 = image2.cuda()
        d1 = model(image1)
        d2 = model(image2)
        pred = model.predict((d1 - d2)**2).squeeze(-1)
        test_loss = criterion(pred, label)
        cls = torch.where(torch.sigmoid(pred) > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
        
        test_losses.update(test_loss.item(), image1.shape[0])
        test_acc.update(torch.mean((cls == label).float()) * 100, image1.shape[0])
    return test_losses.avg, test_acc.avg

def test2(model, test_loader):
    image = test_loader.dataset.tensors[0]
    label = test_loader.dataset.tensors[2]
    pred = DCN(model(image.cuda()).squeeze(-1))
    slope, intercept, r_value, p_value, std_err = sc.stats.linregress(pred, DCN(label))
    return r_value, slope

def test3(model, slope, test_loader):
    mean_var = AverageMeter("TestMeanVar", ":.4e") 
    for image1, image2, answer1, answer2, label in test_loader:
        label = label.cuda()
        image1 = image1.cuda()
        d1 = model(image1)
        mean_var.update(torch.std(d1 * slope).item())
    return mean_var.avg


def main():
    tracemalloc.start()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameter check

    print(f'system : {args.system}')
    print(f'iter : {args.iter}')
    print(f'n : {args.n}')
    print(f'm : {args.m}')
    print(f'noise : {args.noise}')
    print(f'indicator : {args.indicator}')

    system_dict = {'S1': system_S1, 'S2': system_S2, 'S3': system_S3, 'P1': system_P1, 'P2': system_P2}
    len_dict = {'S1': (4, 0), 'S2':(3, 0), 'S3': (4, 0), 'P1': (2, 0), 'P2': (4, 0) }

    formula_len = len_dict[args.system][0]
    noise_len = len_dict[args.system][1]
    
    system_name = system_dict[args.system]
    rule_name = args.model + '_' + args.system
    total_size = args.n * args.m * 2
    batch_size = args.m * 2
    batch_num = int(total_size / batch_size)
    print(total_size, batch_size, batch_num)
    train_ratio = 0.5
    noise = args.noise

    generator = DataGen(system_name, batch_size, batch_num)
    file_name = rule_name + '_L' + str(formula_len) + '_N' + str(noise_len) + '_B' + str(batch_num) + '_n' + str(noise)

    if not os.path.isfile('./data/' + file_name + '_train.pkl'):
        generator.run(file_name, total_size, batch_size, train_ratio, noise_strength=noise)

    # Loader

    with open('./data/' + file_name + '_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./data/' + file_name + '_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    noise_var = args.noise
    train_shape = torch.FloatTensor(train_data['Image1']).shape
    test_shape = torch.FloatTensor(test_data['Image1']).shape
    tmax = torch.ones(formula_len + noise_len)
    if args.system == 'P1':
        tmax = torch.FloatTensor([10., 10.])
    elif args.system == 'P2':
        tmax = torch.FloatTensor([10., 10., 1., 1.])

    if args.system == 'P2':
        train_data = TensorDataset(torch.FloatTensor(train_data['Image1']) / tmax + noise_var * torch.randn(*train_shape),
                                torch.FloatTensor(train_data['Image2']) / tmax + noise_var * torch.randn(*train_shape),
                                torch.FloatTensor(train_data['Answer1'])[:, 0],
                                torch.FloatTensor(train_data['Answer2'])[:, 0],
                                torch.FloatTensor(train_data['Label']))

        test_data = TensorDataset(torch.FloatTensor(test_data['Image1']) / tmax + noise_var * torch.randn(*test_shape),
                                torch.FloatTensor(test_data['Image2']) / tmax + noise_var * torch.randn(*test_shape),
                                torch.FloatTensor(test_data['Answer1'])[:, 0],
                                torch.FloatTensor(test_data['Answer2'])[:, 0],
                                torch.FloatTensor(test_data['Label']))
    else:
        train_data = TensorDataset(torch.FloatTensor(train_data['Image1']) / tmax + noise_var * torch.randn(*train_shape),
                            torch.FloatTensor(train_data['Image2']) / tmax + noise_var * torch.randn(*train_shape),
                            torch.FloatTensor(train_data['Answer1']),
                            torch.FloatTensor(train_data['Answer2']),
                            torch.FloatTensor(train_data['Label']))

        test_data = TensorDataset(torch.FloatTensor(test_data['Image1']) / tmax + noise_var * torch.randn(*test_shape),
                            torch.FloatTensor(test_data['Image2']) / tmax + noise_var * torch.randn(*test_shape),
                            torch.FloatTensor(test_data['Answer1']),
                            torch.FloatTensor(test_data['Answer2']),
                            torch.FloatTensor(test_data['Label']))

    train_loader = DataLoader(
                train_data,
                batch_size=64,
                shuffle=True,
                pin_memory=True,
                num_workers=args.workers,
            )
    test_loader = DataLoader(
                test_data,
                batch_size=64,
                shuffle=True,
                pin_memory=True,
                num_workers=args.workers,
            )

    test_loader2 = DataLoader(
                test_data,
                batch_size=int((1 - train_ratio) * batch_size),
                shuffle=False,
                pin_memory=True,
            )

    for i in range(formula_len + noise_len):
        print('x{} : min = {}, max = {}'.format(i, min(train_data.tensors[0][:, i]), max(train_data.tensors[0][:,i])))
    print(f'C : min = {min(train_data.tensors[2])}, max = {max(train_data.tensors[2])}')
    # Spreader
    
    # corr

    D_in = formula_len + noise_len
    D_hidden = 320
    D_out = 1
    cfg_clf = [D_in, D_hidden, D_hidden, D_hidden, D_hidden, D_out]
    model_list = []

    for iter in range(args.iter):
        model = SiameseNet(cfg_clf, 'mlp', 1).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []
        mv_list = []
        corr_list = []
        best_loss = np.inf
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(0, args.epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer)
            test_loss, test_acc = test(model, test_loader, criterion)
            corr, slope = test2(model, test_loader)
            mean_var = test3(model, slope, test_loader2)
            is_best = test_loss < best_loss
            best_corr = min(test_loss, best_loss)
            
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            mv_list.append(mean_var)
            corr_list.append(np.abs(corr))
            
            if is_best:
                best_model = model
        
        model_list.append({
                "epoch": epoch,
                "model_state_dict": best_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
                "MV" : mean_var,
                "best_loss" : best_corr,
                "train_loss_list" : train_loss_list,
                "test_loss_list" : test_loss_list,
                "train_acc_list" : train_acc_list,
                "test_acc_list" : test_acc_list,
                "mv_list" : mv_list,
                "corr_list" : corr_list
            })

    with open('./result/' + file_name + args.indicator + '.pkl', 'wb') as f:
            pickle.dump(model_list, f)

if __name__ == "__main__":
    print("started!")  # For test
    main()
