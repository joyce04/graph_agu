code=train_base.py
dataset=citeseer
config=./configs/base.json
gnn_type="gcn"
python $code --dataset $dataset --config $config --gnn $gnn_type --data_split public
python $code --dataset $dataset --config $config --gnn $gnn_type --data_split full

gnn_type="graphsage"
python $code --dataset $dataset --config $config --gnn $gnn_type --data_split public
python $code --dataset $dataset --config $config --gnn $gnn_type --data_split full

gnn_type="gat"
python $code --dataset $dataset --config $config --gnn $gnn_type --data_split public
python $code --dataset $dataset --config $config --gnn $gnn_type --data_split full
