#!/bin/bash


# for lr in 1e-3
# do
#     for nl in 1 2
#     do
#         for cl in 1024 2048 4096 8192 16384
#         do
#             python train.py \
#                 --gpu_id=2 \
#                 --data_dir=./SoC_synthesis \
#                 --lr=$lr \
#                 --save_path=/data/jan4021/ESS_generation/lr$lr
#         done
#     done
# done


# for lr in 1e-3 1e-4 1e-5
# do
#     for l1 in 0.01 0.05 0.1
#     do
#         for nl in 1 2
#         do
#             for dropout in 0.01 0.05 0.1

#         python train.py \
#             --gpu_id=0 \
#             --data_dir=./data/SoC_synthesis3 \
#             --lambda1=$l1 --lr=$lr \
#             --save_path=./data/train_result/lr$lr\_lambda1_$l1\_
#     done
# done

for lr in 1e-4 1e-5 1e-6
do
    for min_lr in 1e-6 1e-7 1e-8 1e-9
    do
        python train.py \
            --gpu_id=0 \
            --data_dir=./data/SoC_synthesis3 \
            --lambda1=0.1 --lr=$lr --min_lr=$min_lr \
            --save_path=./data/train_result/lr$lr\_min_lr$min_lr
    done
done
