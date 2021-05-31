code=train_base.py
dataset=cora
config=./configs/gaug.json
gnn_type="gcn"
gaug_type="O"
python $code --config $config --gnn $gnn_type --data_split public --gaug_type $gaug_type --removal_rate 10 --add_rate 60
python $code --config $config --gnn $gnn_type --data_split full --gaug_type $gaug_type --removal_rate 10 --add_rate 60

gnn_type="graphsage"
gaug_type="O"
python $code --config $config --gnn $gnn_type --data_split public --gaug_type $gaug_type --removal_rate 10 --add_rate 60
python $code --config $config --gnn $gnn_type --data_split full --gaug_type $gaug_type --removal_rate 10 --add_rate 60

gnn_type="gat"
gaug_type="O"
python $code --config $config --gnn $gnn_type --data_split public --gaug_type $gaug_type --removal_rate 10 --add_rate 60
python $code --config $config --gnn $gnn_type --data_split full --gaug_type $gaug_type --removal_rate 10 --add_rate 60