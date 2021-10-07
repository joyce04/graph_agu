ython train_fb_base.py --config ./configs/base.json --gnn fbgcn --dataset citeseer --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/flag.json --gnn fbgcn --dataset citeseer --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/grand.json --gnn fbgcn --dataset citeseer --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_group.json --gnn fbgcn --dataset citeseer --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset citeseer --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset citeseer --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train.py --config ./configs/flag_orig.json --gnn gcn --dataset citeseer --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train.py --config ./configs/flag_orig.json --gnn gcn --dataset citeseer --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1

python train_fb_base.py --config ./configs/base.json --gnn fbgcn --dataset squirrel --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/flag.json --gnn fbgcn --dataset squirrel --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb_base.py --config ./configs/grand.json --gnn fbgcn --dataset squirrel --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_group.json --gnn fbgcn --dataset squirrel --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset squirrel --data_split public --epochs 150 --patience 10 --learning_rate 0.003 --cr 1
python train_fb.py --config ./configs/flag_orig.json --gnn fbgcn --dataset squirrel --data_split public --epochs 150 --patience 10 --learning_rate 0.003
python train.py --config ./configs/flag_group.json --gnn gcn --dataset squirrel --data_split public --epochs 150 --patience 10 --learning_rate 0.003