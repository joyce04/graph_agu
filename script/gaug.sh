tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=train_base.py
dataset=cora
config=./configs/gaug.json
gnn_type="gcn"
gaug_type="O"
edge_split=1
gint=0
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.01 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 42 --add_rate 52 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.03 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 48 --add_rate 69 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.05 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 72 --add_rate 87 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.1 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 16 --add_rate 55 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.2 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 17 --add_rate 96 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.3 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 14 --add_rate 60 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.4 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 30 --add_rate 53 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.5 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 14 --add_rate 74 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --edge_split $edge_split --gnn $gnn_type --train_ratio 0.6 --gaug_type $gaug_type --gaug_interval $gint --removal_rate 25 --add_rate 88 & wait" C-m

edge_split=0--gaug_interval $gint
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.01 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 10 --add_rate 92 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.03 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 26 --add_rate 76 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.05 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 11 --add_rate 66 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.1 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 20 --add_rate 55 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.2 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 24 --add_rate 51 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.3 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 42 --add_rate 83 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.4 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 11 --add_rate 72 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.5 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 16 --add_rate 71 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --train_ratio 0.6 --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 13 --add_rate 52 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --data_split public --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 12 --add_rate 87 & wait" C-m
tmux send-keys "python $code --config $config --dataset $dataset --gnn $gnn_type --data_split full --edge_split $edge_split --gaug_type $gaug_type --gaug_interval $gint --removal_rate 31 --add_rate 69 & wait" C-m