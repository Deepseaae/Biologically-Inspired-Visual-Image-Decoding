#!/bin/bash
# cd /home/xxxxxxxx/Biologically-Inspired-Visual-Image-Decoding

# retrieval assignment
brain_backbone="EEG_Encoder"
vision_backbone="VIT-H-14"

dataset="eeg"
i="08"
seed=0

#python3 main.py --config configs/eeg/baseline.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
# python3 main.py --config configs/eeg/baseline.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

dataset="meg"
i="08"
seed=0

#python3 main.py --config configs/meg/baseline.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting intra-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-4;
# python3 main.py --config configs/meg/baseline.yaml --dataset $dataset --subjects sub-$i --seed $seed --exp_setting inter-subject --brain_backbone $brain_backbone --vision_backbone $vision_backbone --epoch 50 --lr 1e-5;

