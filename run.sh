#!/bin/bash
python train_fb_base.py --config ./configs/base.json --gnn fbgat --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/flag.json --gnn fbgat --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_orig.json --gnn fbgat --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_orig.json --gnn fbgat --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1

python train_base.py --config ./configs/base.json --gnn gat --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_base.py --config ./configs/flag.json --gnn gat --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train.py --config ./configs/flag_orig.json --gnn gat --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train.py --config ./configs/flag_orig.json --gnn gat --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1
