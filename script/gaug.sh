tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=train_base.py
dataset=cora
config=./configs/gaug.json
gnn_type="gcn"
gaug_type="O"
edge_split=1
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.01 --gaug_type $gaug_type --removal_rate 23 --add_rate 51 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.03 --gaug_type $gaug_type --removal_rate 85 --add_rate 14 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.05 --gaug_type $gaug_type --removal_rate 71 --add_rate 49 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.1 --gaug_type $gaug_type --removal_rate 43 --add_rate 17 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.2 --gaug_type $gaug_type --removal_rate 79 --add_rate 15 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.3 --gaug_type $gaug_type --removal_rate 12 --add_rate 17 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.4 --gaug_type $gaug_type --removal_rate 42 --add_rate 65 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.5 --gaug_type $gaug_type --removal_rate 31 --add_rate 63 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.6 --gaug_type $gaug_type --removal_rate 35 --add_rate 12 & wait" C-m
#
#edge_split=0
#tmux send-keys "python $code --config $config --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
#tmux send-keys "python $code --config $config --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
#tmux send-keys "python $code --config $config --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
#tmux send-keys "python $code --config $config --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
#tmux send-keys "python $code --config $config --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
#tmux send-keys "python $code --config $config --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
#tmux send-keys "python $code --config $config --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
#tmux send-keys "python $code --config $config --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
#tmux send-keys "python $code --config $config --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --gaug_type $gaug_type --removal_rate 37 --add_rate 43 --gaug_interval 20 & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --gaug_type $gaug_type --removal_rate 30 --add_rate 70 --gaug_interval 30 & wait" C-m

gnn_type="gcn"