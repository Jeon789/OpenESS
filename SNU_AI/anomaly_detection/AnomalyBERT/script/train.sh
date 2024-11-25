"""
Anomaly Detection
작업 위치 : /home/ess/year345/Anomaly_Detection
데이터 관련 config 위치 : ./utils/config.py



"""
# 훈련(시온유 Dataset)
python train.py --dataset ESS_sionyu --gpu_id 1

# 전이학습
# finetune(시온유 -> 판리)
python train.py --dataset ESS_panli --gpu_id 3 --checkpoint logs/ESS_sionyu/state_dict.pt --lr 0.00001 --max_steps 50000

# Anomaly Score 및 (NiN+FiF)/2 계산(state_dict의 dataset으로 test)
python estimate.py --gpu_id 1 --model logs/ESS_sionyu/model.pt --state_dict logs/ESS_sionyu/state_dict.pt
