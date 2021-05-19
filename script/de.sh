tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=train_base.py
dataset=cora
config=./configs/de.json
tmux send-keys "python $code --config $config --train_ratio 0.01 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.03 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.05 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.1 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.2 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.3 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.4 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.5 & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.6 & wait" C-m

edge_split=0
tmux send-keys "python $code --config $config --train_ratio 0.01 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.03 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.05 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.1 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.2 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.3 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.4 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.5 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --train_ratio 0.6 --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --data_split public --edge_split $edge_split & wait" C-m
tmux send-keys "python $code --config $config --data_split full --edge_split $edge_split & wait" C-m