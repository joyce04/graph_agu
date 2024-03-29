# params determined by optuna optimization
tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=train.py
dataset=cora
config=./configs/flag_orig_gaug.json
gnn_type="gcn"
cr=0
edge_split=1
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr & wait" C-m

cr=1
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr & wait" C-m

edge_split=0
cr=0
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --cr $cr & wait" C-m

cr=1
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --cr $cr & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --cr $cr & wait" C-m
#code=train_base.py
#dataset=cora
#config=./configs/flag.json
#gnn_type="gcn"
#cr=0
#edge_split=1
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr --m 5 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr --m 5 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr --m 3 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr --m 4 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr --m 4 & wait" C-m
#
#cr=1
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr --m 5 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr --m 4 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr --m 5 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr --m 4 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr --m 3 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr --m 5 & wait" C-m
#
#edge_split=0
#cr=0
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr --m 5 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr --m 3 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr --m 4 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --cr $cr --m 3 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --cr $cr --m 3 & wait" C-m
#
#cr=1
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr --m 4 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr --m 3 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr --m 4 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr --m 3 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr --m 2 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --cr $cr --m 3 & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --cr $cr --m 3 & wait" C-m