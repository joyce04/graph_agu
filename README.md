### Data Augmentation Approaches for Graph Neural Networks

##### Highlights
- Review recent data augmentation approaches for GNNs
- Experiments with following baseline models: GCN, GAT, GraphSage
- Data augmentation studied:  
    - Node-level augmentation : FLAG[1] [git repository](https://github.com/devnkong/FLAG)
    - Topology-level augmentation : G-Aug[2] [git repository](https://github.com/zhao-tong/GAug)
    - Topology-level updates : DropEdge[3] [git repository](https://github.com/DropEdge/DropEdge)

<br>

[optuna](https://github.com/optuna/optuna) is used to find best parameters

<br>

##### Requirements
- Python 3.8.2
- optuna==2.7.0
- torch==1.8.0
- torch-geometric==1.6.3
- torch-scatter==2.0.6
- torch-sparse==0.6.9
Please refer to the requirements.txt for any other packages.
<br>

##### Implementation Details
All GNNs are implemented with pytorch-geometric.
<br>

##### Usage
To run the demo :
- For baseline gcn, graphsage, gat : sh ./script/base_trial.sh
- For FLAG gcn, graphsage, gat : sh ./script/flag_trial.sh
- For DropEdge gcn, graphsage, gat : sh ./script/de_trial.sh
- For G-Aug gcn, graphsage, gat : sh ./script/de_trial.sh
Please modify json files in ./configs to change any hyperparameters.


##### References
[1] Kong, Kezhi, et al. "Flag: Adversarial data augmentation for graph neural networks." arXiv preprint arXiv:2010.09891 (2020).<br>
[2] Zhao, Tong, et al. "Data Augmentation for Graph Neural Networks." arXiv preprint arXiv:2006.06830 (2020).
[3] Rong, Yu, et al. "Dropedge: Towards deep graph convolutional networks on node classification." arXiv preprint arXiv:1907.10903 (2019).


[comment]: <> (# TODO)

[comment]: <> (experimental result comparison :)

[comment]: <> (- https://paperswithcode.com/sota/node-classification-on-cora-with-public-split)

[comment]: <> (- https://paperswithcode.com/sota/node-classification-on-cora-full-supervised)

[comment]: <> (# check best parameter for GAT : )

[comment]: <> (- https://arxiv.org/pdf/1710.10903v3.pdf)