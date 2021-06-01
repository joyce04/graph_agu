code=train_base.py
config=./configs/de.json
gnn_type=gcn
python $code --config $config --data_split public --gnn $gnn_type --dataset "cora" --de_sampling_percent 0.7
python $code --config $config --data_split full --gnn $gnn_type --dataset "cora" --de_sampling_percent 0.7
python $code --config $config --data_split public --gnn $gnn_type --dataset "citeseer" --de_sampling_percent 0.05
python $code --config $config --data_split full --gnn $gnn_type --dataset "citeseer" --de_sampling_percent 0.05
python $code --config $config --data_split public --gnn $gnn_type --dataset "pubmed" --de_sampling_percent 0.3
python $code --config $config --data_split full --gnn $gnn_type --dataset "pubmed" --de_sampling_percent 0.3

gnn_type=graphsage
python $code --config $config --data_split public --gnn $gnn_type --dataset "cora" --de_sampling_percent 0.4
python $code --config $config --data_split full --gnn $gnn_type --dataset "cora" --de_sampling_percent 0.4
python $code --config $config --data_split public --gnn $gnn_type --dataset "citeseer" --de_sampling_percent 0.1
python $code --config $config --data_split full --gnn $gnn_type --dataset "citeseer" --de_sampling_percent 0.1
python $code --config $config --data_split public --gnn $gnn_type --dataset "pubmed" --de_sampling_percent 0.8
python $code --config $config --data_split full --gnn $gnn_type --dataset "pubmed" --de_sampling_percent 0.8

gnn_type=gat
python $code --config $config --data_split public --gnn $gnn_type --dataset "cora" --de_sampling_percent 0.7
python $code --config $config --data_split full --gnn $gnn_type --dataset "cora" --de_sampling_percent 0.7
python $code --config $config --data_split public --gnn $gnn_type --dataset "citeseer" --de_sampling_percent 0.05
python $code --config $config --data_split full --gnn $gnn_type --dataset "citeseer" --de_sampling_percent 0.05
python $code --config $config --data_split public --gnn $gnn_type --dataset "pubmed" --de_sampling_percent 0.8
python $code --config $config --data_split full --gnn $gnn_type --dataset "pubmed" --de_sampling_percent 0.8