tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=train_base.py
config=./configs/de.json
gnn_type=gcn
tmux send-keys "python $code --config $config --train_ratio 0.01 --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.03 --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.05 --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.1 --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.2 --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.3 --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.4 --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.5 --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.6 --gnn $gnn_type & wait" C-m

edge_split=0
tmux send-keys "python $code --config $config --train_ratio 0.01 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.03 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.05 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.1 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.2 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.3 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.4 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.5 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.6 --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --data_split public --edge_split $edge_split --gnn $gnn_type & wait" C-m
tmux send-keys "python $code --config $config --data_split full --edge_split $edge_split --gnn $gnn_type & wait" C-m