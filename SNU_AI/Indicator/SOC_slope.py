import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import sys

def SOC_slope(folder_path): 
    # 폴더 내의 모든 parquet 파일을 찾기
    
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    slope_value = []
    # 5개 열로 나열할 수 있도록 플롯을 설정
    fig, axs = plt.subplots(len(parquet_files) // 5 + 1, 5, figsize=(25, (len(parquet_files) // 5 + 1) * 5))

    for idx, file in enumerate(parquet_files):
        file_path = os.path.join(folder_path, file)
        
        # parquet 파일 읽기
        # df = pd.read_csv(file_path)
        df = pd.read_parquet(file_path)
        
        #방전구간 찾기
        discharge_mask = df['SOC'].diff() < 0  # SOC가 감소하는 구간을 필터링
        if discharge_mask.sum() == 0:
            print(f"No discharge found in file: {file}")
            continue
        
        discharge_stamp = df[discharge_mask].index

        first_discharge_index = discharge_stamp[0]
        last_discharge_index = discharge_stamp[-1]

        discharge_soc = df.loc[first_discharge_index:last_discharge_index, 'SOC']
        discharge_time = df.loc[first_discharge_index:last_discharge_index].index

        n_segments = 10
        segment_length = len(discharge_soc) // n_segments + 1  ##마지막 segment 짧은구간 생기는 것 방지
        segments = [discharge_soc[i:i+segment_length] for i in range(0, len(discharge_soc), segment_length)]
        time_segments = [discharge_time[i:i+segment_length] for i in range(0, len(discharge_time), segment_length)]

        slopes = []
        for soc_segment, time_segment in zip(segments, time_segments):
            if len(soc_segment) > 1:  # 구간에 데이터가 있는 경우에만
                time_difference = time_segment[-1] - time_segment[0]
                slope = (soc_segment.iloc[-1] - soc_segment.iloc[0]) / time_difference
                slopes.append(slope)
                
        max_slope = min(slopes) * 10
        # breakpoint()
        
        row, col = divmod(idx, 5)
        axs[row, col].plot(df.index, df['SOC'], label='SOC', color='red')
        axs[row, col].set_title(f'File: {file}', fontsize=15)
        axs[row, col].annotate(f"Max Slope: {max_slope:.4f}", xy=(0.5, 0.5), xycoords='axes fraction', fontsize=15,
                               bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('SOC')
        slope_value.append(max_slope)
        
    plt.tight_layout()
    tensor_slopes = torch.tensor(slope_value)
    mean_value = torch.mean(tensor_slopes)
    print(f'평균 : {mean_value}')
    
    # 그래프 보여주기
    plt.savefig('plot')
    
# folder_path = '/data/sky3alfory/HCT_100cycle_detach/sliced/overcharge_parquet'
folder_path = sys.argv[1]
SOC_slope(folder_path)