code=train_base.py
dataset=cora
config=./configs/base.json
gnn_type="gcn"
python $code --config $config --gnn $gnn_type --data_split public
python $code --config $config --gnn $gnn_type --data_split full

gnn_type="graphsage"
python $code --config $config --gnn $gnn_type --data_split public
python $code --config $config --gnn $gnn_type --data_split full

gnn_type="gat"
python $code --config $config --gnn $gnn_type --data_split public
python $code --config $config --gnn $gnn_type --data_split full
