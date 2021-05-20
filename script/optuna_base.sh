tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=optuna_optimize.py
dataset=cora
#config=./configs/flag.json
#gnn_type="gcn"
#cr=0
#edge_split=1
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr & wait" C-m
#
#cr=1
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --cr $cr & wait" C-m
#
#edge_split=0
#cr=0
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --cr $cr & wait" C-m
#
#cr=1
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --cr $cr & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --cr $cr & wait" C-m

config=./configs/gaug.json
edge_split=1
gnn_type="gcn"
gaug_type="O"
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m

edge_split=0
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --gaug_type $gaug_type & wait" C-m

gnn_type="graphsage"
gaug_type="O"
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m

edge_split=0
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --gaug_type $gaug_type & wait" C-m

gnn_type="gat"
gaug_type="O"
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m

edge_split=0
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --gaug_type $gaug_type & wait" C-m


#config=./configs/de.json
#edge_split=1
#gnn_type="gcn"
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split & wait" C-m
#
#edge_split=0
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split & wait" C-m
#
#gnn_type="graphsage"
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split & wait" C-m
#
#edge_split=0
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split & wait" C-m
#
#gnn_type="gat"
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split & wait" C-m
#
#edge_split=0
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split & wait" C-m
#tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split & wait" C-m