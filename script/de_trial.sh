code=train_base.py
config=./configs/de.json
gnn_type=gcn
python $code --config $config --data_split public
python $code --config $config --data_split full

gnn_type=graphsage
python $code --config $config --data_split public
python $code --config $config --data_split full

gnn_type=gat
python $code --config $config --data_split public
python $code --config $config --data_split full