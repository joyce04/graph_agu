tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=optuna_optimize.py
dataset=cora
config=./configs/gaug.json
gnn_type=gcn
gaug_type=O

edge_split=1
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m

edge_split=0
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --data_split public --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --data_split full --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
