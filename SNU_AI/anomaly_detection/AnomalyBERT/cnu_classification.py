"""
AnomalyBERT
################################################

Reference:
    Yungi Jeong et al. "AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme" in ICLR Workshop, "Machine Learning for Internet of Things(IoT): Datasets, Perception, and Understanding" 2023.

Reference:
    https://github.com/Jhryu30/AnomalyBERT
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import argparse

import torch.nn.functional
from tqdm import tqdm

from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

import utils.config as config
from models.anomaly_transformer import get_classification_anomaly_transformer

from estimate import estimate
from compute_metrics import ess_score


from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math




def write_log(log_file, str, mode='a'):
    with open(log_file, mode) as f:
        f.write(str)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def adjust_learning_rate(optimizer, epoch, initial_lr, total_epoch, style):
    """Decay the learning rate based on schedule"""
    lr = initial_lr
    if style =='cosine':  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / total_epoch))
    else:  # stepwise lr schedule
        for milestone in [10,20,30,40,50,60,70,80,90]:
            lr *= 0.5 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"lr = {lr}")


class CNU_Dataset(Dataset):
    def __init__(self, npy, option, transforms=None, mode='train'):
        super().__init__()
        self.npy  = np.load(npy)
        self.transforms = transforms
        self.mode = mode
        self.n_features = option.n_features
        self.patch_size = option.patch_size


        if mode == 'train' : assert 'test' not in npy
        elif mode == 'test': assert 'train' not in npy

        # convert to df and give column name for usability
        COL_NAME = ['V', 'I', 'SOC', 'T', 'dV', 'N', 'label']
        self.df = pd.DataFrame(self.npy)
        self.df.columns = COL_NAME

        self.cycle_nums = self.df['N'].unique()

    def __getitem__(self, index):
        
        # get index = cycle number (N)
        cycle_num = self.cycle_nums[index]
        cycle = self.df[self.df['N']==cycle_num]

        # to tensor
        cycle = torch.tensor(cycle.values)
        
        # cut the long rest at front if the index >= 86400
        cycle = cycle[-86400:, :]

        # split data(5 columns) & label
        data, label = cycle[:,:5], cycle[:, 6]

        # random pick (n_feature x patch_size)
        length = self.n_features * self.patch_size
        if data.size()[0] <= length:
            breakpoint()
        start = np.random.choice(range(0,data.size()[0]-length))
        data = data[start:start+length]


        if self.transforms:
            data = self.transforms(data)
        assert len(label.unique()) == 1

        # TODO not fancy code
        # label = torch.tensor(int(label.unique().item()))
        # label = torch.nn.functional.one_hot(label, num_classes=3)  #  3 is for normal, overcharge, overdischarge
        label = label.unique().item()
        

        # return data.type(torch.float32), label.type(torch.float32)
        return data.type(torch.float32), label
    
    def __len__(self):
        return len(self.cycle_nums)


class CNU_Dataloader(DataLoader):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __len__(self):
        return len(self.dataset)


def main(options):

    train_npy = '/data/ess/data/incell/anomaly_data/data4ESS/datasets/processed/cnu_train.npy'
    test_npy  = '/data/ess/data/incell/anomaly_data/data4ESS/datasets/processed/cnu_test.npy'
    train_data = CNU_Dataset( train_npy ,options, mode='train')
    test_data  = CNU_Dataset( test_npy  ,options, mode='test')
    # train_dataloader = CNU_Dataloader(train_data)
    # test_dataloader = CNU_Dataloader(test_data)

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_dataloader  = DataLoader(test_data,  batch_size=4)

    

    num_label = 3 # normal/overcharge/overdischarge
    d_data = 5 #V,I,SOC,T,dV
    n_features = options.n_features

    # Define model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = get_classification_anomaly_transformer(input_d_data=d_data,
                                    output_d_data=num_label,
                                    patch_size=options.patch_size,
                                    d_embed=options.d_embed,
                                    hidden_dim_rate=4.,
                                    max_seq_len=n_features,
                                    positional_encoding=None,
                                    relative_position_embedding=True,
                                    transformer_n_layer=options.n_layer,
                                    transformer_n_head=8,
                                    dropout=options.dropout).to(device)
    
    # Load a checkpoint if exists.
    if options.checkpoint != None:
        try:
            model.load_state_dict(torch.load(options.checkpoint, map_location='cpu'), strict=False)
        except:
            loaded_weight = torch.load(options.checkpoint, map_location='cpu')
            loaded_weight['linear_embedding.weight'] = model.linear_embedding.weight
            loaded_weight['linear_embedding.bias'] = model.linear_embedding.bias
            model.load_state_dict(loaded_weight)

    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    # log_dir = os.path.join(config.LOG_DIR, time.strftime('%y%m%d%H%M%S_'+options.dataset, time.localtime(time.time())))
    log_folder = options.dataset if options.outfolder == None else options.outfolder
    log_dir = os.path.join(config.LOG_DIR, log_folder)
    
    if os.path.exists(log_dir):
        for index in range(10000):
            if os.path.exists(log_dir + '_' + str(index)):
                continue
            else:
                log_dir = log_dir + '_' + str(index)
                break

    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'state'))
    
    # hyperparameters save
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)
    
    summary_writer = SummaryWriter(log_dir)
    torch.save(model, os.path.join(log_dir, 'model.pt'))

    # Train model.
    max_iters = options.max_steps + 1

    # Train loss
    lr = options.lr
    ce_loss = nn.CrossEntropyLoss()
    # sigmoid = nn.Sigmoid().to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = CosineLRScheduler(optimizer,
    #                               t_initial=max_iters,
    #                               lr_min=lr*0.01,
    #                               warmup_lr_init=lr*0.001,
    #                               warmup_t=max_iters // 10,
    #                               cycle_limit=1,
    #                               t_in_epochs=False,
    #                              )
    
    # Start training.
    for i in tqdm(range(options.initial_iter, max_iters)):
        model.train()
        adjust_learning_rate(optimizer, epoch=i, initial_lr=options.lr, total_epoch=max_iters, style='cosine')
        for data, labels in train_dataloader:

            data, labels = data.to(device), labels.to(device)
            labels = labels.type(torch.cuda.LongTensor)

            y = model(data)
            # Compute losses.
            loss = ce_loss(y, labels)
            

        if i % options.summary_steps == 0:
            # with torch.no_grad():
            summary_writer.add_scalar('Train/Loss', loss.item(), i)  
            torch.save(model.state_dict(), os.path.join(log_dir, 'state/state_dict_step_{}.pt'.format(i)))

        # Update gradients.
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip_norm)

        optimizer.step()
        # scheduler.step_update(i)

        torch.save(model.state_dict(), os.path.join(log_dir, 'state_dict.pt'))


        # Test
        correct = 0
        total = 0
        test_epochs = 100
        test_freq = 10
        if (i+1) % test_freq == 0 :        
            classes = ['normal', 'overcharge', 'overdischarge']
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            with torch.no_grad():
                for _ in range(test_epochs):
                    model.eval()
                    for data, labels in test_dataloader:
                        data, labels = data.to(device), labels.to(device)
                        labels = labels.type(torch.cuda.LongTensor)

                        y = model(data)
                        _, pred = torch.max(y, dim=-1)
                        total += labels.size(0)
                        correct += (pred == labels).sum().item()

                        for label, prediction in zip(labels, pred):
                            if label == prediction:
                                correct_pred[classes[label]] += 1
                            total_pred[classes[label]] += 1

            # print & log message
            msg = ''
            msg += f'Epoch:{i} test accuracy for {test_epochs} test_epoch : {100 * correct / total:3.0f}------------------------- \n'
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                msg += f'Accuracy for class: {classname:5s} is {accuracy:.1f} % \n'
            print(msg)

            log_file = os.path.join(log_dir, 'log.txt') 
            write_log(log_file, msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=2, type=int)
    parser.add_argument("--test", default=False, type=str2bool)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_steps", default=100, type=int, help='maximum_training_steps')
    parser.add_argument("--summary_steps", default=1, type=int, help='steps for summarizing and saving of training log')
    parser.add_argument("--checkpoint", default=None, type=str, help='load checkpoint file')
    parser.add_argument("--initial_iter", default=0, type=int, help='initial iteration for training')
    parser.add_argument("--schedule_style", default='cosine', type=str)

    
    parser.add_argument("--dataset", default='ESS_CNU_classification_JJH', type=str) ##TODO 다 하고 나면 JJH는 삭제해야함
    parser.add_argument("--train_data", default='/data/ess/data/incell/anomaly_data/data4ESS/datasets/processed/cnu_train.npy', type=str)
    parser.add_argument("--test_data", default='/data/ess/data/incell/anomaly_data/data4ESS/datasets/processed/cnu_test.npy', type=str)
    
    parser.add_argument("--n_features", default=512, type=int, help='number of features for a window')
    parser.add_argument("--patch_size", default=90, type=int, help='number of data points in a patch')
    parser.add_argument("--d_embed", default=512, type=int, help='embedding dimension of feature')
    parser.add_argument("--n_layer", default=6, type=int, help='number of transformer layers')
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--window_sliding", default=512, type=int, help='sliding steps of windows for validation')
    
    
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)

    parser.add_argument("--outfolder", default=None, type=str)
    
    # parser.add_argument("--default_options", default=None, type=str, help='default options for datasets; None(default)/SMAP/MSL/SMD/SWaT/WADI')
    
    options = parser.parse_args()
    # if options.default_options != None:
    #     if options.default_options.startswith('SMD'):
    #         default_options = options.default_options
    #         options = torch.load('data/default_options_SMD.pt')
    #         options.dataset = default_options
    #     else:
    #         options = torch.load('data/default_options_'+options.default_options+'.pt')
    
    main(options)