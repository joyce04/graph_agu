#!/bin/bash

python train_fb_base.py --config ./configs/base.json --gnn fbgcn --dataset cora --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/flag.json --gnn fbgcn --dataset cora --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/grand.json --gnn fbgcn --dataset cora --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_group.json --gnn fbgcn --dataset cora --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset cora --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset cora --data_split public --epochs 150 --patience 10 --learning_rate 0.003

python train_fb_base.py --config ./configs/base.json --gnn fbgcn --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/flag.json --gnn fbgcn --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/grand.json --gnn fbgcn --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_group.json --gnn fbgcn --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset chameleon --data_split public --epochs 150 --patience 10 --learning_rate 0.003

python train_fb_base.py --config ./configs/base.json --gnn fbgcn --dataset cornell --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/flag.json --gnn fbgcn --dataset cornell --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/grand.json --gnn fbgcn --dataset cornell --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_group.json --gnn fbgcn --dataset cornell --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset cornell --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset cornell --data_split public --epochs 150 --patience 10 --learning_rate 0.003

python train_fb_base.py --config ./configs/base.json --gnn fbgcn --dataset texas --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/flag.json --gnn fbgcn --dataset texas --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/grand.json --gnn fbgcn --dataset texas --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_group.json --gnn fbgcn --dataset texas --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset texas --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset texas --data_split public --epochs 150 --patience 10 --learning_rate 0.003