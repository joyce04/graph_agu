tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=train_base.py
dataset=cora
config=./configs/gaug.json
gnn_type="gcn"
gaug_type="O"
edge_split=1
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.01 --gaug_type $gaug_type --removal_rate 71 --add_rate 125 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.03 --gaug_type $gaug_type --removal_rate 79 --add_rate 51 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.05 --gaug_type $gaug_type --removal_rate 34 --add_rate 68 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.1 --gaug_type $gaug_type --removal_rate 61 --add_rate 108 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.2 --gaug_type $gaug_type --removal_rate 30 --add_rate 59 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.3 --gaug_type $gaug_type --removal_rate 12 --add_rate 82 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.4 --gaug_type $gaug_type --removal_rate 21 --add_rate 69 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.5 --gaug_type $gaug_type --removal_rate 11 --add_rate 118 & wait" C-m
tmux send-keys "python $code --config $config --edge_split $edge_split --gnn $gnn_type --train_ratio 0.6 --gaug_type $gaug_type --removal_rate 17 --add_rate 133 & wait" C-m

edge_split=0
tmux send-keys "python $code --config $config --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 20 --add_rate 118 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 29 --add_rate 144 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 59 --add_rate 134 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 13 --add_rate 59 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 24 --add_rate 51 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 12 --add_rate 52 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 46 --add_rate 74 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 12 --add_rate 51 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type --removal_rate 13 --add_rate 107 & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split public --edge_split $edge_split --gaug_type $gaug_type --removal_rate 31 --add_rate 109 & wait" C-m
tmux send-keys "python $code --config $config --gnn $gnn_type --data_split full --edge_split $edge_split --gaug_type $gaug_type --removal_rate 30 --add_rate 91 & wait" C-m