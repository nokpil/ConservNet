# coding=utf-8
import argparse
import os
import numpy as np
import scipy as sc

import torch
import torch.nn as nn
import torch.nn.parallel
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
 
from src.utils import *
from src.system import *

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
    "--model", "--model", default="Con", type=str, help="simulation data type : Con(servNet), Siam"
)
parser.add_argument(
    "--system", default="S1", type=str, help="simulation sytem, S1, S2, S3"
)
parser.add_argument(
    "--spreader", default="L2", type=str, help="spreader, L1, L2, L8"
)

parser.add_argument("--iter", default=10, type=int, help="iter num")

parser.add_argument("--n", default=10, type=int, help="group num")
parser.add_argument("--m", default=200, type=int, help="data num")

parser.add_argument("--Q", default=1., type=float, help="Spreader constant")
parser.add_argument("--constant", default=1., type=float, help="Noise norm")
parser.add_argument("--beta", default=1., type=float, help="Variance term constant")
parser.add_argument("--noise", default=0., type=float, help="noise strength")

parser.add_argument(
    "--indicator", default="", type=str, help="Additional specification for file name."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed for torch and numpy")

class ConservNet(nn.Module):
    def __init__(self, cfg_clf, block_type, D_agent):
        super(ConservNet, self).__init__()
        self.classifier = cfg_Block(block_type, cfg_clf, D_agent, 'MS', False, False)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
                #m.weight.data.fill_(0.1)
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.classifier(x)
        return out

# Standard
# Train / Test set shares CONSTANT 

class DataGen():
    def __init__(self, system_type, batch_size, batch_num):
        self.system = system_type(batch_size, batch_num)
    
    def run(self, file_name, total_size, batch_size, train_ratio, noise_strength=0):

        train_image = []
        train_label = []
        test_image = []
        test_label = []
        
        batch_num = int(total_size / batch_size)
        
        for i in range(batch_num):
            for j in range(batch_size):
                data, answer = next(self.system)
                if j < batch_size * train_ratio:
                    train_image.append(data.astype(float))
                    train_label.append(answer)
                else:
                    test_image.append(data.astype(float))
                    test_label.append(answer)
        
        train_output = {'Image': train_image, 'Label': train_label}
        test_output = {'Image': test_image, 'Label': test_label}
        
        if noise_strength > 0:
            print("구현중")
            #ti = np.array(train_image)
            #for i in range(len(ti.shape[0])):
                
        # Output pickle
        with open('./data/' + file_name + '_train.pkl', 'wb') as f:
            pickle.dump(train_output, f)

        with open('./data/' + file_name + '_test.pkl', 'wb') as f:
            pickle.dump(test_output, f)
            

##########
# Plugin #
##########

def plugin_L1(image, constant=0.1):
    size = image.shape
    n_vectors = size[0]
    d = size[1]
    c = constant
    rnd_vec = np.random.uniform(-1, 1, size=(n_vectors, d))
    rnd_vec = rnd_vec / np.expand_dims(np.sum(np.abs(rnd_vec), axis=1), axis=1)
    rnd_vec = rnd_vec * np.expand_dims(np.random.uniform(c, c, n_vectors), axis=1)
    return torch.FloatTensor(rnd_vec)

def plugin_L2(image, constant=np.sqrt(0.1)):
    size = image.shape
    n_vectors = size[0]
    d = size[1]
    c = np.sqrt(constant)
    rnd_vec = np.random.uniform(-1, 1, size=(n_vectors, d))                # the initial random vectors
    unif = np.random.uniform(size=n_vectors)                               # a second array random numbers
    scale_f = np.expand_dims(np.linalg.norm(rnd_vec, axis=1) / unif, axis=1) / c # the scaling factors
    rnd_vec = rnd_vec / scale_f
    return torch.FloatTensor(rnd_vec)

def plugin2_L1(image, alpha=0.5, constant=np.sqrt(0.1)):
    size = image.shape
    mean = np.mean(DCN(image), axis=0)
    n_vectors = size[0]
    d = size[1]
    c = constant
    rnd_vec = np.random.uniform(-1, 1, size=(n_vectors, d))
    rnd_vec = rnd_vec / np.expand_dims(np.sum(np.abs(rnd_vec), axis=1), axis=1)
    rnd_vec = rnd_vec * np.expand_dims(np.random.uniform(c, c, n_vectors), axis=1)
    return torch.FloatTensor(rnd_vec) * (mean * alpha)

def plugin2_L2(image, alpha=0.5, constant=np.sqrt(0.1)):
    size = image.shape
    mean = np.mean(DCN(image), axis=0)
    n_vectors = size[0]
    d = size[1]
    c = np.sqrt(constant)
    rnd_vec = np.random.uniform(-1, 1, size=(n_vectors, d))                # the initial random vectors
    unif = np.random.uniform(size=n_vectors)                               # a second array random numbers
    scale_f = np.expand_dims(np.linalg.norm(rnd_vec, axis=1) / unif, axis=1) / c # the scaling factors
    rnd_vec = (rnd_vec / scale_f) * (mean * alpha)
    return torch.FloatTensor(rnd_vec)

def plugin_L8(image, constant=0.1):
    size = image.shape
    n_vectors = size[0]
    d = size[1]
    c = constant
    rnd_vec = np.random.uniform(-1, 1, size=(n_vectors, d)) * c
    return torch.FloatTensor(rnd_vec)

class spreader():
    def __init__(self, spreader_type, constant=0.1):
        self.spreader_type = spreader_type
        self.constant = constant
        
    def generate(self, image):
        if self.spreader_type == 'L1':
            return plugin_L1(image, self.constant)
        elif self.spreader_type == 'L2':
            return plugin_L2(image, self.constant)
        elif self.spreader_type == 'L8':
            return plugin_L8(image, self.constant)
        elif self.spreader_type == 'L12':
            return plugin2_L1(image, 0.5, self.constant)
        elif self.spreader_type == 'L22':
            return plugin2_L2(image, 0.5, self.constant)
        else:
            raise NotImplementedError
    
def train(model, train_loader, optimizer, plugin, epoch, Q, beta):
    train_losses = AverageMeter("TrainLoss", ":.4e")
    #Q = Q*max(1-epoch/(1000), 0.1)
    for image, label in train_loader:
        label = label.cuda()
        image = image.cuda()
        d1 = model(image) 
        d2 = model(image + plugin.generate(image).cuda())
        train_loss = torch.var(d1) + beta * torch.abs(Q - torch.var(d2, dim=0))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses.update(train_loss.item(), image.shape[0])
    return train_losses.avg

def test(model, test_loader, plugin, epoch, Q, beta):
    test_losses = AverageMeter("TestLoss", ":.4e")
    mean_var = AverageMeter("MeanVar", ":.4e") 
    
    image = test_loader.dataset.tensors[0]
    label = test_loader.dataset.tensors[1]

    pred = DCN(model(image.cuda()).squeeze(-1))
    slope, intercept, r_value, p_value, std_err = sc.stats.linregress(pred, DCN(label))
    
    for image, label in test_loader:
        label = label.cuda()
        image = image.cuda()
        d1 = model(image) 
        d2 = model(image + plugin.generate(image).cuda())
        test_loss = torch.var(d1) + beta * torch.abs(Q - torch.var(d2))
        test_losses.update(test_loss.item(), image.shape[0])
        mean_var.update(torch.std(d1 * slope).item())
        
    return test_losses.avg, r_value, mean_var.avg


def main():
    tracemalloc.start()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameter check

    print(f'system : {args.system}')
    print(f'spreader : {args.spreader}')
    print(f'iter : {args.iter}')
    print(f'n : {args.n}')
    print(f'm : {args.m}')
    print(f'Q : {args.Q}')
    print(f'constant : {args.constant}')
    print(f'beta : {args.beta}')
    print(f'noise : {args.noise}')
    print(f'indicator : {args.indicator}')

    system_dict = {'S1': system_S1, 'S2': system_S2, 'S3': system_S3, 'P1': system_P1, 'P2': system_P2}
    len_dict = {'S1': (4, 0), 'S2':(3, 0), 'S3': (4, 0), 'P1': (2, 0), 'P2': (4, 0) }

    formula_len = len_dict[args.system][0]
    noise_len = len_dict[args.system][1]

    system_name = system_dict[args.system]
    rule_name = args.model + '_' + args.system
    batch_num = args.n
    # batch_size = int(2000 / batch_num) * 2
    batch_size = args.m * 2
    total_size = args.n * batch_size
    batch_num = int(total_size / batch_size)
    print(total_size, batch_size, batch_num)
    train_ratio = 0.5
    noise = args.noise

    generator = DataGen(system_name, batch_size, batch_num)
    file_name = rule_name + '_L' + str(formula_len) + '_N' + str(noise_len) +'_B' + str(batch_num) + '_n' + str(noise)

    if not os.path.isfile('./data/' + file_name + '_train.pkl'):
        generator.run(file_name, total_size, batch_size, train_ratio, noise_strength=noise)

    # Loader

    with open('./data/' + file_name + '_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./data/' + file_name + '_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    noise_var = args.noise
    train_shape = torch.FloatTensor(train_data['Image']).shape
    test_shape = torch.FloatTensor(test_data['Image']).shape
    tmax = torch.ones(formula_len + noise_len)
    print(train_shape, test_shape, tmax)
    if args.system == 'P1':
        tmax = torch.FloatTensor([10., 10.])
    elif args.system == 'P2':
        tmax = torch.FloatTensor([10., 10., 1., 1.])
    if args.system == 'P2':
        train_data = TensorDataset(torch.FloatTensor(train_data['Image']) / tmax + noise_var * torch.randn(*train_shape),
                                torch.FloatTensor(train_data['Label'])[:, 0])

        test_data = TensorDataset(torch.FloatTensor(test_data['Image']) / tmax + noise_var * torch.randn(*train_shape),
                                torch.FloatTensor(test_data['Label'])[:, 0])
    else:
        train_data = TensorDataset(torch.FloatTensor(train_data['Image']) / tmax + noise_var * torch.randn(*train_shape),
                            torch.FloatTensor(train_data['Label']))

        test_data = TensorDataset(torch.FloatTensor(test_data['Image']) / tmax + noise_var * torch.randn(*train_shape),
                                torch.FloatTensor(test_data['Label']))

    train_loader = DataLoader(
                train_data,
                batch_size=int(train_ratio * batch_size),
                shuffle=False,
                pin_memory=True,
                num_workers=args.workers
            )
    test_loader = DataLoader(
                test_data,
                batch_size=int((1 - train_ratio) * batch_size),
                shuffle=False,
                pin_memory=True,
                num_workers=args.workers
            )

    for i in range(formula_len + noise_len):
        print('x{} : min = {}, max = {}'.format(i, min(train_data.tensors[0][:,i]), max(train_data.tensors[0][:,i])))
    #print(f'C : min = {min(train_data.tensors[1][:,0])}, max = {max(train_data.tensors[1][:,0])}')
    print(f'C : min = {min(train_data.tensors[1])}, max = {max(train_data.tensors[1])}')

    # Spreader
   
    D_in = formula_len + noise_len
    D_hidden = 320
    D_out = 1
    cfg_clf = [D_in, D_hidden, D_hidden, D_hidden, D_hidden, D_out]
    model_list = []
    indicator = args.indicator
    
    for iter in range(args.iter):
        model = ConservNet(cfg_clf, 'mlp', 1).cuda()
        train_loss_list = []
        test_loss_list = []
        mv_list = []
        corr_list = []
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
        best_loss = np.inf
        plugin = spreader('L2', args.constant)
        Q = args.Q
        beta = args.beta
        best_model = None
       
        for epoch in range(0, args.epochs):
            train_loss = train(model, train_loader, optimizer, plugin, epoch, Q, beta)
            test_loss, corr, mean_var = test(model, test_loader, plugin, epoch, Q, beta)
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            mv_list.append(mean_var)
            corr_list.append(np.abs(corr))

            if is_best:
                best_model = model
            
        model_list.append({
                        "epoch": epoch,
                        "model_state_dict": best_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": test_loss,
                        "MV": mean_var,
                        "best_loss": best_loss,
                        "train_loss_list": train_loss_list,
                        "test_loss_list": test_loss_list,
                        "mv_list": mv_list,
                        "corr_list": corr_list})
    
    with open('./result/' + file_name + indicator + '.pkl', 'wb') as f:
        pickle.dump(model_list, f)

if __name__ == "__main__":
    print("started!")  # For test
    main()
