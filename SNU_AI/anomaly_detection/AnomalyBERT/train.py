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
from torch.utils.data import DataLoader
import argparse

from tqdm import tqdm

from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

import utils.config as config
from models.anomaly_transformer import get_anomaly_transformer
from utils.dataset import ESS_dataset

from estimate import estimate
from compute_metrics import ess_score



def main(options):
    if options.checkpoint is not None: # for transfer learning
        prev_file = torch.load(options.checkpoint, map_location='cpu')
        prev_options = prev_file['options']
        assert options.dataset != prev_options.dataset, 'Only for transfer learning, not resuming.'

        options.n_features = prev_options.n_features
        options.patch_size = prev_options.patch_size
        options.d_embed = prev_options.d_embed
        options.n_layer = prev_options.n_layer
        options.dropout = prev_options.dropout

    # Load data.
    train_dataset = ESS_dataset(options=options)
    test_dataset = ESS_dataset(options=options, seed=0)
    if options.deg_num is None:
        options.deg_num = test_dataset.time_len // 86400
    test_data, test_label = test_dataset.get_test_data(deg_num=options.deg_num)

    # Define model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = get_anomaly_transformer(input_d_data=train_dataset.column_len,
                                    output_d_data=1,
                                    patch_size=options.patch_size,
                                    d_embed=options.d_embed,
                                    hidden_dim_rate=4.,
                                    max_seq_len=options.n_features,
                                    positional_encoding=None,
                                    relative_position_embedding=True,
                                    transformer_n_layer=options.n_layer,
                                    transformer_n_head=8,
                                    dropout=options.dropout).to(device)
    
    # Load a checkpoint if exists.
    if options.checkpoint is not None:
        loaded_weight = prev_file['model_state_dict']
        try:
            model.load_state_dict(loaded_weight)
        except:
            loaded_weight['linear_embedding.weight'] = model.linear_embedding.weight
            loaded_weight['linear_embedding.bias'] = model.linear_embedding.bias
            model.load_state_dict(loaded_weight)

    if not os.path.exists(config.LOG_DIR):
        os.mkdir(config.LOG_DIR)
    log_dir = os.path.join(config.LOG_DIR, time.strftime('%y%m%d%H%M%S_'+options.dataset, time.localtime(time.time())))
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'state'))
    
    # hyperparameters save
    with open(os.path.join(log_dir, 'hyperparameters.txt'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)
    
    summary_writer = SummaryWriter(log_dir)
    torch.save(model, os.path.join(log_dir, 'model.pt'))

    # Train loss
    train_loss = nn.BCELoss().to(device)
    sigmoid = nn.Sigmoid().to(device)

    # Optimizer and scheduler
    max_iters = options.max_steps + 1
    lr = options.lr
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=max_iters,
                                  lr_min=lr*0.01,
                                  warmup_lr_init=lr*0.001,
                                  warmup_t=max_iters // 10,
                                  cycle_limit=1,
                                  t_in_epochs=False,
                                 )
    
    # Start training.
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=options.batch_size)

    for i, batch in enumerate(tqdm(train_dataloader)):
        x, x_anomaly, _= batch

        # Process data.
        x = x.to(device)
        x_anomaly = x_anomaly.to(device)
        y = model(x).squeeze(-1)

        # Compute losses.
        loss = train_loss(sigmoid(y), x_anomaly)

        # Print training summary.
        if i % options.summary_steps == 0:
            with torch.no_grad():
                n_batch = options.batch_size
                pred = (sigmoid(y) > 0.5).int()
                x_anomaly = x_anomaly.bool().int()
                total_data_num = n_batch * train_dataset.data_seq_len
                
                acc = (pred == x_anomaly).int().sum() / total_data_num
                summary_writer.add_scalar('Train/Loss', loss.item(), i)
                summary_writer.add_scalar('Train/Accuracy', acc, i)
                
                model.eval()

                estimation = estimate(test_data, model, sigmoid, 1, n_batch, options.window_sliding, None, device)
                estimation = estimation[:, 0].cpu().numpy()
                model.train()
                
                best_eval = (0, 0, 0)
                best_rate = 0
                for rate in np.arange(0.001, 0.301, 0.001):
                    evaluation = ess_score(test_label, estimation, rate, False, False)
                    if evaluation[2] > best_eval[2]:
                        best_eval = evaluation
                        best_rate = rate
                summary_writer.add_scalar('Valid/Best Anomaly Rate', best_rate, i)
                summary_writer.add_scalar('Valid/Precision', best_eval[0], i)
                summary_writer.add_scalar('Valid/Recall', best_eval[1], i)
                summary_writer.add_scalar('Valid/ess', best_eval[2], i)
                
                print(f'iteration: {i} | loss: {loss.item():.10f} | train accuracy: {acc:.10f}')
                print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')

            torch.save({'model_state_dict' : model.state_dict(), 'options' : options}, os.path.join(log_dir, 'state/state_dict_step_{}.pt'.format(i)))

        # Update gradients.
        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip_norm)

        optimizer.step()
        scheduler.step_update(i)

    torch.save({'model_state_dict' : model.state_dict(), 'options' : options}, os.path.join(log_dir, 'state_dict.pt'))
    print(log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--max_steps", default=150000, type=int, help='maximum_training_steps')
    parser.add_argument("--summary_steps", default=500, type=int, help='steps for summarizing and saving of training log')
    parser.add_argument("--checkpoint", default=None, type=str, help='load checkpoint file')
    parser.add_argument("--initial_iter", default=0, type=int, help='initial iteration for training')
    
    parser.add_argument("--dataset", default='ESS_panli', type=str, help='ESS_panli/ESS_gold/ESS_white')
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--n_features", default=512, type=int, help='number of features for a window')
    parser.add_argument("--patch_size", default=90, type=int, help='number of data points in a patch')
    parser.add_argument("--d_embed", default=512, type=int, help='embedding dimension of feature')
    parser.add_argument("--n_layer", default=6, type=int, help='number of transformer layers')
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--replacing_rate_max", default=0.15, type=float, help='maximum ratio of replacing interval length to window size')
    
    parser.add_argument("--soft_replacing", default=0.5, type=float, help='probability for soft replacement')
    parser.add_argument("--uniform_replacing", default=0.15, type=float, help='probability for uniform replacement')
    parser.add_argument("--peak_noising", default=0.15, type=float, help='probability for peak noise')
    # parser.add_argument("--length_adjusting", default=0.0, type=float, help='probability for length adjustment')
    parser.add_argument("--white_noising", default=0.0, type=float, help='probability for white noise (deprecated)')
    parser.add_argument("--voltage_gap", default=0.0, type=float, help='probability for voltage gap degradation')
    parser.add_argument("--deg_num", default=None, type=int, help='the number of degradation in test data')
    
    parser.add_argument("--flip_replacing_interval", default='all', type=str,
                        help='allowance for random flipping in soft replacement; vertical/horizontal/all/none')
    parser.add_argument("--replacing_weight", default=0.7, type=float, help='weight for external interval in soft replacement')
    
    parser.add_argument("--window_sliding", default=512, type=int, help='sliding steps of windows for validation')
    
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)

    parser.add_argument("--outfolder", default=None, type=str)
     
    options = parser.parse_args()

    main(options)