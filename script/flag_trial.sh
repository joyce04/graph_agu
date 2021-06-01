code=train_base.py
dataset=citeseer
config=./configs/flag.json
gnn_type="gcn"
biased=0
python $code --config $config --gnn $gnn_type --data_split public --dataset $dataset --biased $biased
python $code --config $config --gnn $gnn_type --data_split full --dataset $dataset --biased $biased

gnn_type="graphsage"
python $code --config $config --gnn $gnn_type --data_split public --dataset $dataset --biased $biased
python $code --config $config --gnn $gnn_type --data_split full --dataset $dataset --biased $biased

gnn_type="gat"
python $code --config $config --gnn $gnn_type --data_split public --dataset $dataset --biased $biased
python $code --config $config --gnn $gnn_type --data_split full --dataset $dataset --biased $biased