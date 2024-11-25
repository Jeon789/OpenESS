import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def select_oneday_data(data, date=None, year=None, month=None, day=None, local_timezone=True):
    """
    data : data for all day
    date : date when data is collected, type=datetime.date or tuple of (year, month, day)
    year, month, day : date when data is collected, used when date is None
    local_timezone : True for local timezone(Asia/Seoul), False for UTC
    """
    if date == None:
        date = datetime.date(year, month, day)
    elif not isinstance(date, datetime.date):
        date = datetime.date(*date)
    timezone = 'Asia/Seoul' if local_timezone else 'UTC'
    return data[pd.to_datetime(data['TIMESTAMP'], utc=True).dt.tz_convert(timezone).dt.date == date]

def generate(
    base_dir = '/data/ess/data/incell/sionyu_old/interpolated_data/monthly/bank/',
    bank_data_file = '220701-220731_bank.parquet',
    seed = 7,
    flag = True
    ):
    bank_data = pd.read_parquet(base_dir + bank_data_file)
    np.random.seed(seed)

    # print(bank_data.head())

    usable_columns = ['TIMESTAMP', 'BANK_DC_VOLT', 'BANK_DC_CURRENT', 'BANK_SOC', 'MAX_CELL_TEMPERATURE_OF_BANK', 'CELL_VOLTAGE_GAP'] #, 'CHARGE_STATUS', 'DISCHARGE_STATUS']

    voltage_gap = bank_data['MAX_CELL_VOLTAGE_OF_BANK'] - bank_data['MIN_CELL_VOLTAGE_OF_BANK']
    # charge_status = (bank_data['BATTERY_STATUS_FOR_CHARGE'] == 2)
    # discharge_status = (bank_data['BATTERY_STATUS_FOR_CHARGE'] == 3)

    bank_data = bank_data[usable_columns[:-1]]
    bank_data[usable_columns[-1]] = voltage_gap

    # bank_data = bank_data[usable_columns[:-3]]
    # bank_data[usable_columns[-3]] = voltage_gap
    # bank_data[usable_columns[-2]] = charge_status
    # bank_data[usable_columns[-1]] = discharge_status

    # Collect data for clear days.
    VOLTAGE_THRES = 790  # 맑은날을 구분하는 볼트 임계값
    thres_above = bank_data[['TIMESTAMP', 'BANK_DC_VOLT']][bank_data['BANK_DC_VOLT'] >= VOLTAGE_THRES]
    clear_date = pd.to_datetime(thres_above['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date.unique()
    whole_date = pd.to_datetime(bank_data['TIMESTAMP'], utc=True).dt.tz_convert('Asia/Seoul').dt.date.unique()

    invalid_date = tuple()   # 전부 다 사용가능한 데이터면 () 빈튜플로 남기고, 못 쓰는 데이터가 있으면 해당일을 튜플 안에 넣을 것
    whole_date = np.delete(whole_date, invalid_date)
    clear_date = np.array([date for date in whole_date if date in clear_date])


    test_clear_date = np.random.permutation(clear_date)[:max(1, int(0.2*len(clear_date)))]
    test_cloudy_date = [date for date in whole_date if date not in clear_date]
    test_cloudy_date = np.random.permutation(test_cloudy_date)[:max(1, int(0.2*len(test_cloudy_date)))]

    test_data = {}
    for date in whole_date:
        if (date in test_clear_date) or (date in test_cloudy_date):
            test_data[date] = select_oneday_data(bank_data, date=date)

    # Configurations for anomaly synthesis
    REPLACING_RATE = (0.0005, 0.1)  # min-max rate of the length of replacing interval for anomalies

    # anomaly synthesis options
    MAX_EXTERNAL_INTERVAL_RATE = 0.7
    MAX_UNIFORM_VALUE_DIFFERENCE = 0.1
    MIN_PEAK_ERROR = 0.1

    # data value ranges
                            # V         # I          # SOC      # temp    # gap
    VALUE_RANGE = np.array([[660, 820], [-280, 410], [0, 100], [10, 40], [0, 0.2]]).transpose()

    # data infos
    N_COLUMNS = 5
    N_ANOMALY_TYPES = 4

    # Synthesize single type of anomaly.
    single_type_data_dir = '/home/ess/year345/Anomaly_Detection/phase_test_data/'

    # total data
    # total_npy_data = bank_data.iloc[:, 1:-2].to_numpy().copy()
    total_npy_data = bank_data.iloc[:, 1:].to_numpy().copy()
    total_npy_data = np.clip((total_npy_data - VALUE_RANGE[[0]]) / (VALUE_RANGE[[1]] - VALUE_RANGE[[0]]), 0, 1)
    total_data_len = len(total_npy_data)

    # meta data and anomaly label data (binary)
    meta_data = []
    anomaly_labels = []

    idx = None
    abnormal_dataset = []
    abnormal_length = 0

    for date, data in tqdm(test_data.items()):
        # Normalize data in [0,1].
        # npy_data = data.iloc[:, 1:-2].to_numpy().copy()
        npy_data = data.iloc[:, 1:].to_numpy().copy()
        npy_data = np.clip((npy_data - VALUE_RANGE[[0]]) / (VALUE_RANGE[[1]] - VALUE_RANGE[[0]]), 0, 1)
        data_len = len(npy_data)
        
        _date = date.strftime('%y%m%d')
        anomaly_label = np.zeros((data_len,1))  # last column for anomaly index 0 or 1
        # additional_data = data.iloc[:, -2:].to_numpy().copy()
        
        # Select replacing and target intervals.
        replacing_length_range = (np.array(REPLACING_RATE) * data_len).astype(int)
        replacing_lengths = np.random.randint(*replacing_length_range, size=N_ANOMALY_TYPES)
        target_indices = np.random.randint(0, data_len-replacing_lengths+1)
        abnormal_length += np.sum(replacing_lengths)
        
        # Select abnormal columns.
        abnormal_columns = np.random.rand(N_ANOMALY_TYPES, N_COLUMNS)
        
        
        # Synthesize anomalies. - soft replacing
        syn_data = npy_data.copy()
        _anomaly_label = anomaly_label.copy()
        interval_len = replacing_lengths[0]
        abnormal_column_idx = abnormal_columns[0] < 0.5
        if not(abnormal_column_idx.any()):
            abnormal_column_idx[np.random.randint(0, N_COLUMNS)] = True
        replacing_column_idx = np.random.choice(N_COLUMNS, size=len(abnormal_column_idx[abnormal_column_idx]), replace=False)
        
        replacing_index = np.random.randint(0, total_data_len-interval_len+1)
        external_interval = total_npy_data[replacing_index:replacing_index+interval_len, replacing_column_idx].copy()
        target_interval = syn_data[target_indices[0]:target_indices[0]+interval_len, abnormal_column_idx].copy()
        
        weights = np.concatenate((np.linspace(0, MAX_EXTERNAL_INTERVAL_RATE, num=interval_len//2),
                                np.linspace(MAX_EXTERNAL_INTERVAL_RATE, 0, num=(interval_len+1)//2)), axis=None)
        syn_data[target_indices[0]:target_indices[0]+interval_len, abnormal_column_idx] = weights[:, None] * external_interval\
                                                                                        + (1 - weights[:, None]) * target_interval
        _anomaly_label[target_indices[0]:target_indices[0]+interval_len] = 1
        # syn_data = np.concatenate((syn_data, additional_data, _anomaly_label), axis=1)
        syn_data = np.concatenate((syn_data, _anomaly_label), axis=1)
        
        # file_name = single_type_data_dir+_date+'_0.npy'
        meta_data.append([_date+'_0', 'replacing', (target_indices[0], target_indices[0]+interval_len),
                        tuple(np.argwhere(abnormal_column_idx).flatten()), (date in test_clear_date)])
        abnormal_dataset.append(syn_data)
        # np.save(file_name, syn_data)

        if flag and abnormal_column_idx[0] and date in test_clear_date:
            idx = len(abnormal_dataset) - 1
            plt.figure(figsize=(15, 5))
            plt.scatter(range(86400), syn_data[:,:5][:, abnormal_column_idx][:,0], s=1, c='r', label='Voltage-synthesized anomaly')
            plt.scatter(range(86400), npy_data[:, abnormal_column_idx][:,0], s=1, c='C0', label='Voltage-original')
            # plt.fill_between(range(86400), 0, 1, where=(_anomaly_label[:,0]==1), color='red', alpha=0.3)
            plt.legend(markerscale=10)
            plt.show()
            flag = False
        
        # Synthesize anomalies. - uniform replacing
        syn_data = npy_data.copy()
        _anomaly_label = anomaly_label.copy()
        interval_len = replacing_lengths[1]
        abnormal_column_idx = abnormal_columns[1] < 0.5
        if not(abnormal_column_idx.any()):
            abnormal_column_idx[np.random.randint(0, N_COLUMNS)] = True
            
        mean_values = syn_data[target_indices[1]:target_indices[1]+interval_len, abnormal_column_idx].mean(axis=0)
        syn_data[target_indices[1]:target_indices[1]+interval_len, abnormal_column_idx]\
            = np.random.uniform(np.maximum(mean_values-MAX_UNIFORM_VALUE_DIFFERENCE, 0),
                                np.minimum(mean_values+MAX_UNIFORM_VALUE_DIFFERENCE, 1))[None, :]
        
        _anomaly_label[target_indices[1]:target_indices[1]+interval_len] = 1
        # syn_data = np.concatenate((syn_data, additional_data, _anomaly_label), axis=1)
        syn_data = np.concatenate((syn_data, _anomaly_label), axis=1)
        
        # file_name = single_type_data_dir+_date+'_1.npy'
        meta_data.append([_date+'_1', 'uniform', (target_indices[1], target_indices[1]+interval_len),
                        tuple(np.argwhere(abnormal_column_idx).flatten()), (date in test_clear_date)])
        abnormal_dataset.append(syn_data)
        # np.save(file_name, syn_data)
        
        
        # Synthesize anomalies. - peak noising
        syn_data = npy_data.copy()
        _anomaly_label = anomaly_label.copy()
        interval_len = replacing_lengths[2]
        abnormal_column_idx = abnormal_columns[2] < 0.5
        if not(abnormal_column_idx.any()):
            abnormal_column_idx[np.random.randint(0, N_COLUMNS)] = True
            
        peak_indices = np.random.randint(target_indices[2], target_indices[2]+interval_len,
                                        size=len(abnormal_column_idx[abnormal_column_idx]))
        peak_values = syn_data[peak_indices, abnormal_column_idx].copy()
        
        peak_errors = np.random.uniform(np.minimum(0, MIN_PEAK_ERROR-peak_values), np.maximum(0, 1-peak_values-MIN_PEAK_ERROR))
        peak_values = peak_values + peak_errors + ((peak_errors > 0).astype(int) * 2 - 1) * MIN_PEAK_ERROR
        syn_data[peak_indices, abnormal_column_idx] = peak_values
        
        _anomaly_label[peak_indices] = 1
        # syn_data = np.concatenate((syn_data, additional_data, _anomaly_label), axis=1)
        syn_data = np.concatenate((syn_data, _anomaly_label), axis=1)
        
        # file_name = single_type_data_dir+_date+'_2.npy'
        meta_data.append([_date+'_2', 'peak', (np.min(peak_indices), np.max(peak_indices)+1),
                        tuple(np.argwhere(abnormal_column_idx).flatten()), (date in test_clear_date)])
        abnormal_dataset.append(syn_data)
        # np.save(file_name, syn_data)

    meta_data = pd.DataFrame(meta_data, columns=['name', 'type', 'abnormal_interval', 'abnormal_column', 'clear_date'])
    meta_data.to_csv(single_type_data_dir+'meta_data_'+_date[:-2]+'.csv')

    # Merge all data.
    # meta_dataset = []
    # abnormal_dataset = []
    # for file in sorted(os.listdir(single_type_data_dir)):
    #     if 'meta' in file:
    #         meta_dataset.append(pd.read_csv(single_type_data_dir+file).iloc[:, 1:])
    #     elif '.npy' in file:
    #         abnormal_dataset.append(np.load(single_type_data_dir+file))
            
    # meta_dataset = pd.concat(meta_dataset)
    abnormal_dataset = np.concatenate(abnormal_dataset, axis=0)
    # rate = abnormal_length/len(abnormal_dataset)

    # meta_dataset.to_csv(single_type_data_dir+'meta_data.csv')
    np.save(single_type_data_dir+f'{bank_data_file.split("_")[0]}.npy', abnormal_dataset)

    return abnormal_dataset, idx
