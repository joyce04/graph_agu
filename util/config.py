import argparse
import json

import torch


def get_configs(args):
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    if args.data_loc is None:
        args.data_loc = config['data_loc']
    if args.dataset is None:
        args.dataset = config['dataset']
    if args.data_split is None:
        args.data_split = config['data_split']
    if args.data_split == 'random':
        if args.train_ratio is None:
            args.train_ratio = config['subset_params']['train_ratio']
        if args.edge_split is None:
            args.edge_split = config['subset_params']['edge_split']

    if args.epochs is None:
        args.epochs = config['epochs']

    if args.gnn is None:
        args.gnn = config['gnn']

    if args.seed is None:
        args.seed = config['seed']

    if args.learning_rate is None:
        args.learning_rate = config['learning_rate']
    if args.weight_decay is None:
        args.weight_decay = config['weight_decay']
    if args.patience is None:
        args.patience = config['patience']

    if args.aug_type is None:
        args.aug_type = config['aug_type']
    if args.aug_type == 'flag':
        if args.m is None:
            args.m = config['flag_params']['m']
        if args.step_size is None:
            args.step_size = config['flag_params']['step_size']
        if args.cr is None:
            args.cr = config['flag_params']['cr']

    return args


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file')
    parser.add_argument('--data_loc', help='dataset file location')
    parser.add_argument('--dataset', help='benchmark dataset : cora, citeseer, pubmed')
    parser.add_argument('--data_split', help='public, full, random')
    parser.add_argument('--train_ratio', type=float, help='train subset ratio, Only applicable for random')
    parser.add_argument('--edge_split', type=bool, help='Only applicable for random')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--gnn', help='gcn, graphsage, gat')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--patience', type=int, help='patience for early stopping')
    parser.add_argument('--aug_type', help='augmentation type')
    parser.add_argument('--m', type=int, help='Number of flag iteration')
    parser.add_argument('--step_size', type=float, help='flag step size')
    parser.add_argument('--cr', type=bool, help='To apply CR to flag augmentation or not')

    args = parser.parse_args()
    args = get_configs(args)
    return args


def device_setup():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        return torch.device('cpu')
