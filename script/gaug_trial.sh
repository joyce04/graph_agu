code=train_base.py
dataset=cora
config=./configs/gaug.json
gnn_type="gcn"
gaug_type="O"
python $code --config $config --gnn $gnn_type --dataset "cora" --data_split public --gaug_type $gaug_type --removal_rate 2 --add_rate 57
python $code --config $config --gnn $gnn_type --dataset "cora" --data_split full --gaug_type $gaug_type --removal_rate 2 --add_rate 57
python $code --config $config --gnn $gnn_type --dataset "citeseer" --data_split public --gaug_type $gaug_type --removal_rate 0 --add_rate 46
python $code --config $config --gnn $gnn_type --dataset "citeseer" --data_split full --gaug_type $gaug_type --removal_rate 10 --add_rate 46
python $code --config $config --gnn $gnn_type --dataset "pubmed" --data_split public --gaug_type $gaug_type --removal_rate 10 --add_rate 46
python $code --config $config --gnn $gnn_type --dataset "pubmed" --data_split full --gaug_type $gaug_type --removal_rate 10 --add_rate 46

gnn_type="graphsage"
gaug_type="O"
python $code --config $config --gnn $gnn_type --dataset "cora" --data_split public --gaug_type $gaug_type --removal_rate 1 --add_rate 80
python $code --config $config --gnn $gnn_type --dataset "cora" --data_split full --gaug_type $gaug_type --removal_rate 1 --add_rate 80
python $code --config $config --gnn $gnn_type --dataset "citeseer" --data_split public --gaug_type $gaug_type --removal_rate 2 --add_rate 37
python $code --config $config --gnn $gnn_type --dataset "citeseer" --data_split full --gaug_type $gaug_type --removal_rate 2 --add_rate 37
python $code --config $config --gnn $gnn_type --dataset "pubmed" --data_split public --gaug_type $gaug_type --removal_rate 10 --add_rate 57
python $code --config $config --gnn $gnn_type --dataset "pubmed" --data_split full --gaug_type $gaug_type --removal_rate 10 --add_rate 57

gnn_type="gat"
gaug_type="O"
python $code --config $config --gnn $gnn_type --dataset "cora" --data_split public --gaug_type $gaug_type --removal_rate 53 --add_rate 68
python $code --config $config --gnn $gnn_type --dataset "cora" --data_split full --gaug_type $gaug_type --removal_rate 20 --add_rate 68
python $code --config $config --gnn $gnn_type --dataset "citeseer" --data_split public --gaug_type $gaug_type --removal_rate 40 --add_rate 68
python $code --config $config --gnn $gnn_type --dataset "citeseer" --data_split full --gaug_type $gaug_type --removal_rate 38 --add_rate 68
python $code --config $config --gnn $gnn_type --dataset "pubmed" --data_split public --gaug_type $gaug_type --removal_rate 10 --add_rate 68
python $code --config $config --gnn $gnn_type --dataset "pubmed" --data_split full --gaug_type $gaug_type --removal_rate 10 --add_rate 60