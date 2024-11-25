import os
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.config as config

class ESS_dataset(Dataset):
    def __init__(self, options, seed=None):
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.options = options
        data_path = config.TRAIN_DATASET[options.dataset] if seed is None else config.TEST_DATASET[options.dataset]
        self.data = np.load(data_path).astype(np.float32) # time x columns
        self.voltage_gap_seq = self.data[:,-1]
        self.data = self.data[:,:-1]
        self.time_len, self.column_len = self.data.shape
        self.data_seq_len = options.n_features * options.patch_size
        self.valid_index_list = np.arange(len(self.data) - self.data_seq_len)
        self.numerical_column = np.arange(self.column_len)
        
        self.replacing_rate_max = options.replacing_rate_max
        self.replacing_weight = options.replacing_weight

        # anomaly probs
        self.voltage_gap_prob = 1 - options.voltage_gap
        self.soft_replacing_prob = self.voltage_gap_prob - options.soft_replacing
        self.uniform_replacing_prob = self.soft_replacing_prob - options.uniform_replacing
        self.peak_noising_prob = self.uniform_replacing_prob - options.peak_noising
        self.white_noising_prob = self.peak_noising_prob - options.white_noising
        
        # flip
        flip_replacing_interval = options.flip_replacing_interval.lower()
        if flip_replacing_interval == 'all':
            self.vertical_flip = True
            self.horizontal_flip = True
        elif flip_replacing_interval == 'vertical':
            self.vertical_flip = True
            self.horizontal_flip = False
        elif flip_replacing_interval == 'horizontal':
            self.vertical_flip = False
            self.horizontal_flip = True
        elif flip_replacing_interval == 'none':
            self.vertical_flip = False
            self.horizontal_flip = False

    def __len__(self):
        if self.seed is None:
            return self.options.batch_size * (self.options.max_steps - self.options.initial_iter)
        else:
            return len(self.data)
    
    def __getitem__(self, index):
        assert self.seed is None, 'It is not a train dataset.'

        first_index = np.random.choice(self.valid_index_list) # model input으로 들어갈 수 있는 index 중 하나 선택
        x = torch.tensor(self.data[first_index:first_index+self.data_seq_len].copy()) # size = (data_seq_len, column_len)
        x_true = x.clone()
        x_anomaly = torch.zeros(self.data_seq_len)

        replacing_length = np.random.randint(int(self.data_seq_len*self.replacing_rate_max/10), int(self.data_seq_len*self.replacing_rate_max)) # replacing length 선택

        target_index = np.random.randint(0, self.data_seq_len-replacing_length+1) # size = (1,), model input 중 replacing 구간 및 처음 index 선택
        replacing_type = np.random.uniform(0., 1.) # replacing type을 정하기 위한 변수
        replacing_dim_numerical = np.random.uniform(0., 1., size=self.column_len) # size = (column_len,)
        replacing_dim_numerical = (replacing_dim_numerical - np.maximum(np.min(replacing_dim_numerical), 0.3)) <= 0.001 # 무조건 True 한개 이상 존재, replacing column 선택 위한 변수

        if replacing_length > 0:
            is_replace = True
            
            # voltage_gap anomaly는 전 구간을 사용하므로 우선적으로 처리함
            if replacing_type > self.voltage_gap_prob:
                # Voltage Gap을 반영한 New Voltage 계산 후, 기존 Voltage와 비교하여 New Voltage Gap 계산함
                newV = x[:,0] + torch.Tensor(self.voltage_gap_seq[target_index : target_index+self.data_seq_len])
                x[:,-1] += torch.abs(newV-x[:,0])
                x_anomaly[:] = 1
                
            else:
                _x = x[target_index:target_index+replacing_length].clone().transpose(0, 1) # size = (column_len, replacing_len)
                replacing_number = sum(replacing_dim_numerical) # replacing column 개수
                target_column_numerical = self.numerical_column[replacing_dim_numerical] # replacing column 선택

                if replacing_type > self.soft_replacing_prob:
                    _x[target_column_numerical] = self._soft_replacing(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                    x_anomaly[target_index:target_index+replacing_length] = 1

                elif replacing_type > self.uniform_replacing_prob:
                    _x[target_column_numerical] = self._uniform_replacing(num=replacing_number)
                    x_anomaly[target_index:target_index+replacing_length] = 1

                elif replacing_type > self.peak_noising_prob:
                    peak_value, peak_index = self._peak_noising(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                    _x[target_column_numerical, peak_index] = peak_value

                    peak_index += target_index
                    target_first = np.maximum(0, peak_index - self.options.patch_size) # patch 안에 존재하는지 확인
                    target_last = peak_index + self.options.patch_size + 1
                    x_anomaly[target_first:target_last] = 1

                elif replacing_type > self.white_noising_prob:
                    _x[target_column_numerical] = self._white_noising(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                    x_anomaly[target_index:target_index+replacing_length] = 1

                else:
                    is_replace = False

                if is_replace:
                    x[target_index:target_index+replacing_length] = _x.transpose(0, 1)
                         
        return x, x_anomaly, x_true

    def _soft_replacing(self, x, num, length):
        replacing_index = np.random.randint(0, self.time_len-length+1, size=self.column_len) # size = (column_len,)
        _x = []
        col_num = np.random.choice(self.numerical_column, size=num)
        flip = np.random.randint(0, 2, size=(num, 2)) > 0.5
        for _col, _rep, _flip in zip(col_num, replacing_index, flip):
            random_interval = self.data[_rep:_rep+length, _col].copy()
            if self.horizontal_flip and _flip[0]:
                random_interval = random_interval[::-1].copy()
            if self.vertical_flip and _flip[1]:
                random_interval = 1 - random_interval
            _x.append(torch.from_numpy(random_interval))
        
        _x = torch.stack(_x)
        warmup_len = length//10
        weights = torch.concat((torch.linspace(0, self.replacing_weight, steps=warmup_len),
                                torch.full(size=(length-2*warmup_len,), fill_value=self.replacing_weight),
                                torch.linspace(self.replacing_weight, 0, steps=warmup_len)), dim=0).float().unsqueeze(0)

        return _x * weights + x * (1-weights)

    def _uniform_replacing(self, num):
        return torch.rand(size=(num, 1))
    
    def _peak_noising(self, x, num, length):
        peak_index = np.random.randint(0, length)
        peak_value = (x[:,peak_index] < 0.5).float()
        peak_value = peak_value + (0.1 * (1 - 2 * peak_value)) * torch.rand(size=(num,))

        return peak_value, peak_index

    def _white_noising(self, x, num, length):
        return (x+torch.normal(mean=0, std=0.003, size=(num, length))).clamp(min=0., max=1.)

    def get_test_data(self, deg_num, save=False):
        """
        Writer : parkis

        Get degraded data and degradation label

        Args:
            deg_num (int) : the number of degradation in test data
            save (bool) : whether save test data and test label or not
        Returns:
            test_data (np.array) : data degraded from original test data which is not degraded
            test_label (np.array) : whether degradation exists or not
        """
        assert self.seed is not None, 'It is not a test dataset.'
        
        test_data = self.data.copy()
        test_label = np.zeros(len(self.data))

        for index in range(deg_num):
            first_index = np.random.choice(self.valid_index_list) # model input으로 들어갈 수 있는 index 중 하나 선택
            x = torch.tensor(test_data[first_index:first_index+self.data_seq_len].copy()) # size = (data_seq_len, column_len)
            x_anomaly = torch.zeros(self.data_seq_len)

            replacing_length = np.random.randint(int(self.data_seq_len*self.replacing_rate_max/10), int(self.data_seq_len*self.replacing_rate_max)) # replacing length 선택

            target_index = np.random.randint(0, self.data_seq_len-replacing_length+1) # size = (1,), model input 중 replacing 구간 및 처음 index 선택
            replacing_type = np.random.uniform(0., 1.) # replacing type을 정하기 위한 변수
            replacing_dim_numerical = np.random.uniform(0., 1., size=self.column_len) # size = (column_len,)
            replacing_dim_numerical = (replacing_dim_numerical - np.maximum(np.min(replacing_dim_numerical), 0.3)) <= 0.001 # 무조건 True 한개 이상 존재, replacing column 선택 위한 변수
            
            if replacing_length > 0:
                is_replace = True
                
                # voltage_gap anomaly는 전 구간을 사용하므로 우선적으로 처리함
                if replacing_type > self.voltage_gap_prob:
                    newV = x[:,0] + torch.Tensor(self.voltage_gap_seq[target_index : target_index+self.data_seq_len])
                    x[:,-1] += torch.abs(newV-x[:,0])
                    x_anomaly[:] = 1
                    test_data[first_index:first_index+self.data_seq_len] = x.numpy()
                    test_label[first_index:first_index+self.data_seq_len] = x_anomaly.numpy()
                    
                else:
                    _x = x[target_index:target_index+replacing_length].clone().transpose(0, 1) # size = (column_len, replacing_len)
                    replacing_number = sum(replacing_dim_numerical) # replacing column 개수
                    target_column_numerical = self.numerical_column[replacing_dim_numerical] # replacing column 선택

                    if replacing_type > self.soft_replacing_prob:
                        _x[target_column_numerical] = self._soft_replacing(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                        x_anomaly[target_index:target_index+replacing_length] = 1

                    elif replacing_type > self.uniform_replacing_prob:
                        _x[target_column_numerical] = self._uniform_replacing(num=replacing_number)
                        x_anomaly[target_index:target_index+replacing_length] = 1

                    elif replacing_type > self.peak_noising_prob:
                        peak_value, peak_index = self._peak_noising(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                        _x[target_column_numerical, peak_index] = peak_value

                        peak_index += target_index
                        target_first = np.maximum(0, peak_index - self.options.patch_size) # patch 안에 존재하는지 확인
                        target_last = peak_index + self.options.patch_size + 1
                        x_anomaly[target_first:target_last] = 1

                    elif replacing_type < self.white_noising_prob:
                        _x[target_column_numerical] = self._white_noising(_x[target_column_numerical], num=replacing_number, length=replacing_length)
                        x_anomaly[target_index:target_index+replacing_length] = 1

                    else:
                        is_replace = False

                    if is_replace:
                        x[target_index:target_index+replacing_length] = _x.transpose(0, 1)
                        test_data[first_index:first_index+self.data_seq_len] = x.numpy()
                        test_label[first_index:first_index+self.data_seq_len] = x_anomaly.numpy()

        test_data = test_data.astype(np.float32)
        test_label = test_label.astype(np.int32)

        if save:
            print('Test data save!')
            np.save(os.path.join(self.options.save_folder, self.options.dataset + '_degraded.npy'), test_data)
            np.save(os.path.join(self.options.save_folder, self.options.dataset + '_degraded_label.npy'), test_label)

        return test_data, test_label