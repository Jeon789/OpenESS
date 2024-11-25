"""
AnomalyBERT
################################################

Reference:
    Yungi Jeong et al. "AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme" in ICLR Workshop, "Machine Learning for Internet of Things(IoT): Datasets, Perception, and Understanding" 2023.

Reference:
    https://github.com/Jhryu30/AnomalyBERT
"""

import os, json
import numpy as np
import torch
import argparse
from tqdm import tqdm
from utils.Stdout_Font import Stdout_Font

# Estimate anomaly scores.
def estimate(test_data, model, post_activation, out_dim, batch_size, window_sliding, divisions,
             check_count=None, device='cpu'):
    # Estimation settings
    window_size = model.max_seq_len * model.patch_size
    assert window_size % window_sliding == 0
    
    n_column = out_dim
    n_batch = batch_size
    batch_sliding = n_batch * window_size
    _batch_sliding = n_batch * window_sliding

    output_values = torch.zeros(len(test_data), n_column, device=device)
    count = 0
    checked_index = np.inf if check_count == None else check_count
    
    # Record output values.
    for division in divisions:
        data_len = division[1] - division[0]
        last_window = data_len - window_size + 1
        _test_data = test_data[division[0]:division[1]]
        _output_values = torch.zeros(data_len, n_column, device=device)
        n_overlap = torch.zeros(data_len, device=device)
    
        with torch.no_grad():
            _first = -batch_sliding
            for first in range(0, last_window-batch_sliding+1, batch_sliding):
                for i in range(first, first+window_size, window_sliding):
                    # Call mini-batch data.
                    x = torch.Tensor(_test_data[i:i+batch_sliding].copy()).reshape(n_batch, window_size, -1).to(device)
                    
                    # Evaludate and record errors.
                    y = post_activation(model(x))
                    _output_values[i:i+batch_sliding] += y.view(-1, n_column)
                    n_overlap[i:i+batch_sliding] += 1

                    count += n_batch

                    if count > checked_index:
                        print(count, 'windows are computed.')
                        checked_index += check_count

                _first = first

            _first += batch_sliding

            for first, last in zip(range(_first, last_window, _batch_sliding),
                                   list(range(_first+_batch_sliding, last_window, _batch_sliding)) + [last_window]):
                # Call mini-batch data.
                x = []
                for i in list(range(first, last-1, window_sliding)) + [last-1]:
                    x.append(torch.Tensor(_test_data[i:i+window_size].copy()))

                # Reconstruct data.
                x = torch.stack(x).to(device)

                # Evaludate and record errors.
                y = post_activation(model(x))
                for i, j in enumerate(list(range(first, last-1, window_sliding)) + [last-1]):
                    _output_values[j:j+window_size] += y[i]
                    n_overlap[j:j+window_size] += 1

                count += n_batch

                if count > checked_index:
                    print(count, 'windows are computed.')
                    checked_index += check_count

            # Compute mean values.
            _output_values = _output_values / n_overlap.unsqueeze(-1)
            
            # Record values for the division.
            output_values[division[0]:division[1]] = _output_values
            
    return output_values

def ess_score(gt, pr, anomaly_rate=0.05, adjust=True, modify=False):
    # get anomaly intervals
    gt_aug = np.concatenate([np.zeros(1), gt, np.zeros(1)]).astype(np.int32)
    gt_diff = gt_aug[1:] - gt_aug[:-1]

    begin = np.where(gt_diff == 1)[0]
    end = np.where(gt_diff == -1)[0]

    intervals = np.stack([begin, end], axis=1)

    # quantile cut
    pa = pr.copy()
    q = np.quantile(pa, 1-anomaly_rate)
    pa = (pa > q).astype(np.int32)
    
    for s, e in intervals:
        interval = slice(s, e)
        if pa[interval].sum() > 0:
            pa[interval] = 1

    # confusion matrix
    TP = (gt * pa).sum()
    TN = ((1 - gt) * (1 - pa)).sum()
    FP = ((1 - gt) * pa).sum()
    FN = (gt * (1 - pa)).sum()

    assert (TP + TN + FP + FN) == len(gt)

    # Compute p, r, ess.
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    ess_score = (precision+recall)/2

    return pa, precision, recall, ess_score

def evaluate(data,
        label,
        model='/data/ess/output/Anomaly_Detection/logs/ESS_sionyu/model.pt',
        state_dict='/data/ess/output/Anomaly_Detection/logs/ESS_sionyu/state_dict.pt',
        batch_size=16,
        window_sliding=512,
        check_count=5000,
        anomaly_rate=0.026,
        gpu_id=0
        ):
    # Load test data.
    test_data = data.copy().astype(np.float32)

    # Load model.
    device = torch.device('cuda:{}'.format(gpu_id))
    model = torch.load(model, map_location=device)
    model.load_state_dict(torch.load(state_dict, map_location='cpu'))
    model.eval()
    
    # Data division
    divisions = [[0, len(test_data)]]

    n_column = 1
    post_activation = torch.nn.Sigmoid().to(device)
            
    # Estimate scores.
    output_values = estimate(test_data, model, post_activation, n_column, batch_size,
                             window_sliding, divisions, check_count, device)
    
    # Save results.
    output_values = output_values.cpu().numpy()
    if output_values.ndim == 2:
        output_values = output_values[:, 0]

    test_label = label.copy().astype(np.int32)

    # maximum = 0
    # anomaly_rate = 0.026
    # for rate in tqdm(np.arange(0.001,0.301,0.001)):
    #     output_result, *evaluation = ess_score(test_label, output_values, rate)
    #     if maximum < evaluation[2]:
    #         maximum = evaluation[2]
    #         anomaly_rate = rate
    
    output_result, *evaluation = ess_score(test_label, output_values, anomaly_rate)

    print('\n')
    print(Stdout_Font.Bold + Stdout_Font.Red + f'ESS-Score: {evaluation[2]:.5f}' + Stdout_Font.Reset + ' | ' +  f'recall: {evaluation[1]:.5f}' +  ' | ' + f'precision: {evaluation[0]:.5f}\n')
    
    return output_result, output_values, evaluation

