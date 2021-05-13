tmux new -s base_aug -d
tmux send-keys "source ~/.bashrc" C-m
tmux send-keys "conda activate ocean_g" C-m

code=train_base.py
dataset=cora
config=./configs/gcn_base.json
tmux send-keys "python $code --config $config & wait" C-m

train_ratio=0.03
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.05
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.1
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.15
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.2
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.25
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.3
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.35
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.4
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.45
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.5
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.55
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m

train_ratio=0.6
tmux send-keys "python $code --config $config --train_ratio $train_ratio & wait" C-m