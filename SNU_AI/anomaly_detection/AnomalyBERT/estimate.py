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
import matplotlib.pyplot as plt

from utils.dataset import ESS_dataset
import utils.config as config

# Exponential weighted moving average
def ewma(series, weighting_factor=0.9):
    current_factor = 1 - weighting_factor
    _ewma = series.copy()
    for i in range(1, len(_ewma)):
        _ewma[i] = _ewma[i-1] * weighting_factor + _ewma[i] * current_factor
    return _ewma


# Get anomaly sequences.
def anomaly_sequence(label):
    anomaly_args = np.argwhere(label).flatten()  # Indices for abnormal points.
    
    # Terms between abnormal invervals
    terms = anomaly_args[1:] - anomaly_args[:-1]
    terms = terms > 1

    # Extract anomaly sequences.
    sequence_args = np.argwhere(terms).flatten() + 1
    sequence_length = list(sequence_args[1:] - sequence_args[:-1])
    sequence_args = list(sequence_args)

    sequence_args.insert(0, 0)
    if len(sequence_args) > 1:
        sequence_length.insert(0, sequence_args[1])
    sequence_length.append(len(anomaly_args) - sequence_args[-1])

    # Get anomaly sequence arguments.
    sequence_args = anomaly_args[sequence_args]
    anomaly_label_seq = np.transpose(np.array((sequence_args, sequence_args + np.array(sequence_length))))
    return anomaly_label_seq, sequence_length


# Interval-dependent point
def interval_dependent_point(sequences, lengths):
    n_intervals = len(sequences)
    n_steps = np.sum(lengths)
    return (n_steps / n_intervals) / lengths


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
    
    # Modified Ess
    if modify:
        gt_seq_args, gt_seq_lens = anomaly_sequence(gt)  # gt anomaly sequence args
        ind_p = interval_dependent_point(gt_seq_args, gt_seq_lens)  # interval-dependent point
        
        # Compute TP and FN.
        TP = 0
        FN = 0
        for _seq, _len, _p in zip(gt_seq_args, gt_seq_lens, ind_p):
            n_tp = pa[_seq[0]:_seq[1]].sum()
            n_fn = _len - n_tp
            TP += n_tp * _p
            FN += n_fn * _p
            
        # Compute TN and FP.
        TN = ((1 - gt) * (1 - pa)).sum()
        FP = ((1 - gt) * pa).sum()

    else:
        # point adjustment
        if adjust:
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
    precision = 0 if TP+FP == 0 else TP / (TP + FP)
    recall = 0 if TP+FN == 0 else TP / (TP + FN)
    ess_score = (precision+recall)/2

    return precision, recall, ess_score

# Estimate anomaly scores.
def estimate(test_data, model, post_activation, out_dim, batch_size, window_sliding, check_count=None, device='cpu'):
    """
    Writer : parkis

    Computes anomaly scores of test data
    """
    # Estimation settings
    window_size = model.max_seq_len * model.patch_size
    assert window_size % window_sliding == 0
    
    n_column = out_dim
    n_batch = batch_size
    batch_sliding = n_batch * window_size
    _batch_sliding = n_batch * window_sliding
    data_len = len(test_data)

    count = 0
    checked_index = np.inf if check_count == None else check_count
    
    # Record output values.
    last_window = data_len - window_size + 1
    output_values = torch.zeros(data_len, n_column, device=device)
    n_overlap = torch.zeros(data_len, device=device)

    with torch.no_grad():
        _first = -batch_sliding
        for first in range(0, last_window-batch_sliding+1, batch_sliding):
            for i in range(first, first+window_size, window_sliding):
                # Call mini-batch data.
                x = torch.Tensor(test_data[i:i+batch_sliding].copy()).reshape(n_batch, window_size, -1).to(device)
                
                # Evaludate and record errors.
                y = post_activation(model(x))
                output_values[i:i+batch_sliding] += y.view(-1, n_column)
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
                x.append(torch.Tensor(test_data[i:i+window_size].copy()))

            # Reconstruct data.
            x = torch.stack(x).to(device)

            # Evaludate and record errors.
            y = post_activation(model(x))
            for i, j in enumerate(list(range(first, last-1, window_sliding)) + [last-1]):
                output_values[j:j+window_size] += y[i]
                n_overlap[j:j+window_size] += 1

            count += n_batch

            if count > checked_index:
                print(count, 'windows are computed.')
                checked_index += check_count

        # Compute mean values.
        output_values = output_values / n_overlap.unsqueeze(-1)
        
    return output_values

def compute(test_data, test_label, output_values, prefix, options):
    """
    Writer : parkis
    
    Compute ess scores from anomaly scores of test data

    Args:
        test_data (np.array) : data degraded from original test data which is not degraded
        test_label (np.array) : whether degradation exists or not
        output_values (np.array) : anomaly scores of test data
        prefix (str) : prefix of save file
    """
    if output_values.ndim == 2:
        output_values = output_values[:, 0]
    
    if options.smooth_scores:
        smoothed_values = ewma(output_values, options.smoothing_weight)
    
    result_file = prefix + '_evaluations.txt'
    result_file = open(result_file, 'w')
        
    # Save test data and output results in figures.
    if options.save_figures:
        data_dim = len(test_data[0])
        save_folder = prefix + '_figures/'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        fig, axs = plt.subplots(data_dim, 1, figsize=(20, data_dim))
        for j in range(data_dim):
            axs[j].plot(test_data[:, j], alpha=0.6)
            axs[j].scatter(np.arange(len(test_data))[test_label==1], test_data[test_label==1, j],
                            c='r', s=1, alpha=0.8)
        fig.savefig(save_folder+'data.jpg', bbox_inches='tight')
        plt.close()
        
        fig, axs = plt.subplots(1, figsize=(20, 5))
        axs.plot(output_values, alpha=0.6)
        axs.scatter(np.arange(len(test_data))[test_label==1], output_values[test_label==1],
                    c='r', s=1, alpha=0.8)
        fig.savefig(save_folder+'score.jpg', bbox_inches='tight')
        plt.close()
        
        if options.smooth_scores:
            fig, axs = plt.subplots(1, figsize=(20, 5))
            axs.plot(smoothed_values[:], alpha=0.6)
            axs.scatter(np.arange(len(test_data))[test_label==1], smoothed_values[test_label==1],
                        c='r', s=1, alpha=0.8)
            fig.savefig(save_folder+'sm.jpg', bbox_inches='tight')
            plt.close()
        
    # Compute ess-scores.
    ess_str = 'Modified ess-score' if options.modified_ess else 'ess-score'

    # ess Without PA
    result_file.write('<'+ess_str+' without point adjustment>\n\n')
    
    best_eval = (0, 0, 0)
    best_rate = 0
    for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
        evaluation = ess_score(test_label, output_values, rate, False, options.modified_ess)
        result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | ess-score: {evaluation[2]:.5f}\n')
        if evaluation[2] > best_eval[2]:
            best_eval = evaluation
            best_rate = rate
    result_file.write('\nBest ess-score\n')
    result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n\n\n')
    print('Best ess-score without point adjustment')
    print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')
    
    # ess With PA
    if not options.modified_ess:
        result_file.write('<ess-score with point adjustment>\n\n')
        
        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
            evaluation = ess_score(test_label, output_values, rate, True)
            result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | ess-score: {evaluation[2]:.5f}\n')
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate
        result_file.write('\nBest ess-score\n')
        result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n\n\n')
        print('Best ess-score with point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')
    
    if options.smooth_scores:
        # ess Without PA
        result_file.write('<'+ess_str+' of smoothed scores without point adjustment>\n\n')
        best_eval = (0, 0, 0)
        best_rate = 0
        for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
            evaluation = ess_score(test_label, smoothed_values, rate, False, options.modified_ess)
            result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | ess-score: {evaluation[2]:.5f}\n')
            if evaluation[2] > best_eval[2]:
                best_eval = evaluation
                best_rate = rate
        result_file.write('\nBest ess-score\n')
        result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n\n\n')
        print('Best ess-score of smoothed scores without point adjustment')
        print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')
        
        # ess With PA
        if not options.modified_ess:
            result_file.write('<ess-score of smoothed scores with point adjustment>\n\n')
            best_eval = (0, 0, 0)
            best_rate = 0
            for rate in np.arange(options.min_anomaly_rate, options.max_anomaly_rate+0.001, 0.001):
                evaluation = ess_score(test_label, smoothed_values, rate, True)
                result_file.write(f'anomaly rate: {rate:.3f} | precision: {evaluation[0]:.5f} | recall: {evaluation[1]:.5f} | ess-score: {evaluation[2]:.5f}\n')
                if evaluation[2] > best_eval[2]:
                    best_eval = evaluation
                    best_rate = rate
            result_file.write('\nBest ess-score\n')
            result_file.write(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n\n\n')
            print('Best ess-score of smoothed scores with point adjustment')
            print(f'anomaly rate: {best_rate:.3f} | precision: {best_eval[0]:.5f} | recall: {best_eval[1]:.5f} | ess-score: {best_eval[2]:.5f}\n')
    
    # Close file.
    result_file.close()

def main(options):
    data_folder = f"seed_{options.seed}_soft_{str(options.soft_replacing).split('.')[-1]}_uni_{str(options.uniform_replacing).split('.')[-1]}_peak_{str(options.peak_noising).split('.')[-1]}_vgap_{str(options.voltage_gap).split('.')[-1]}"
    save_folder = os.path.join(options.base_folder, data_folder)
    if os.path.exists(save_folder):
        print("Already exists!")
        exit()
    else:
        os.mkdir(save_folder)

    # Load model.
    device = torch.device('cuda:{}'.format(options.gpu_id))
    model = torch.load(options.model, map_location=device)

    train_file = torch.load(options.state_dict, map_location='cpu')
    test_options = train_file['options']
    test_options.save_folder = save_folder
    test_options.soft_replacing = options.soft_replacing
    test_options.uniform_replacing = options.uniform_replacing
    test_options.volatge_gap = options.voltage_gap
    model.load_state_dict(train_file['model_state_dict'])
    model.eval()

    deg_num = options.deg_num if options.deg_num is not None else test_options.deg_num
    test_dataset = ESS_dataset(options=test_options, seed=options.seed)
    test_data, test_label = test_dataset.get_test_data(deg_num=deg_num, save=options.save_data)

    n_column = 1
    post_activation = torch.nn.Sigmoid().to(device)
            
    # Estimate scores.
    output_values = estimate(test_data, model, post_activation, n_column, test_options.batch_size,
                             test_options.window_sliding, options.check_count, device)
    
    # Save results.
    output_values = output_values.cpu().numpy()
    outfile = os.path.join(save_folder, 'results.npy')
    np.save(outfile, output_values)
    
    compute(test_data, test_label, output_values, outfile[:-4], options)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    
    parser.add_argument("--base_folder", required=True, type=str, help='log folder to estimate')
    parser.add_argument("--model", default=None, type=str, help='model file (.pt) to estimate')
    parser.add_argument("--state_dict", default=None, type=str, help='state dict file (.pt) to estimate')
    parser.add_argument('--save_figures', default=False, action='store_true', help='save figures of data and anomaly scores')
    
    parser.add_argument("--save_data", default=False, action='store_true', help='save test data')
    parser.add_argument("--check_count", default=5000, type=int, help='check count of window computing')
    
    parser.add_argument('--deg_num', default=None, type=int, help='the number of degradation in test data')
    parser.add_argument('--smooth_scores', default=False, action='store_true', help='option for smoothing scores (ewma)')
    parser.add_argument("--smoothing_weight", default=0.9, type=float, help='ewma weight when smoothing socres')
    parser.add_argument('--modified_ess', default=False, action='store_true', help='modified ess scores (not used now)')
    parser.add_argument("--min_anomaly_rate", default=0.001, type=float, help='minimum threshold rate')
    parser.add_argument("--max_anomaly_rate", default=0.3, type=float, help='maximum threshold rate')

    parser.add_argument("--soft_replacing", default=0.3, type=float, help='probability for soft replacement')
    parser.add_argument("--uniform_replacing", default=0.3, type=float, help='probability for uniform replacement')
    parser.add_argument("--peak_noising", default=0.0, type=float, help='probability for peak noise')
    # parser.add_argument("--length_adjusting", default=0.0, type=float, help='probability for length adjustment')
    parser.add_argument("--white_noising", default=0.0, type=float, help='probability for white noise (deprecated)')
    parser.add_argument("--voltage_gap", default=0.4, type=float, help='probability for voltage gap degradation')
    
    options = parser.parse_args()
    if options.model is None:
        options.model = os.path.join(options.base_folder, 'model.pt')
    else:
        options.model = os.path.join(options.base_folder, options.model)
    
    if options.state_dict is None:
        options.state_dict = os.path.join(options.base_folder, 'state_dict.pt')
    else:
        options.state_dict = os.path.join(options.base_folder, options_state_dict)

    import time, datetime
    start = time.time()
    main(options)
    end = time.time()
    print('Time : ' + str(datetime.timedelta(seconds=end-start)).split(".")[0])
