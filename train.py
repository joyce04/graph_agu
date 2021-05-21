import random

import numpy as np
import torch
from torch.optim import Adam

from eval import validate, evaluate
from flag.flag_orig import apply_flag_orig
from gnn.clf import generate_node_clf
from util.config import get_arguments, device_setup
from util.data import dataset_split
from util.tool import EarlyStopping


def train_flag_orig(data, model, optimizer, device, params):
    forward = lambda perturb: model(data.x + perturb, data.train_index)
    model_forward = (model, forward)

    loss = apply_flag_orig(model_forward, data, optimizer, device, params)
    return loss


if __name__ == '__main__':
    args = get_arguments()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_setup()

    with open('./results/nc_{}_{}_{}_{}_es_{}.csv'.format(args.aug_type, args.gnn, args.epochs, args.dataset, str(args.edge_split)), 'a+') as file:
        file.write(','.join(map(lambda x: x + ':' + str(vars(args)[x]), vars(args).keys())) + '\n')
        file.write('run, train F1 avg, validation F1 avg, test F1 avg\n')

        val_f1_list, test_f1_list, train_f1_list = [], [], []

        for r in range(10):
            dataset, data = dataset_split(args.data_loc, args.dataset, args.data_split, args.train_ratio, args.edge_split)
            num_nodes = data.x.shape[0]
            num_feats = data.x.shape[1]
            num_nd_classes = np.max(data.y.numpy()) + 1

            data.x = data.x.to(device)
            model = generate_node_clf(args.gnn, num_feats, num_nd_classes, device)
            model.reset_parameters()
            optimizer = Adam(model.gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            best_test, best_val, best_tr = 0, 0, 0
            lowest_val_loss = float("inf")

            # if args.config.find('gaug.json') >= 0:
            #     if args.gaug_type == 'M':
            #         gaug = GAug(True)
            #         gaug.get_pretrained_edges(data, args.m_file_loc, args.removal_rate, args.add_rate)
            #     else:
            #         gaug = GAug(False)
            #         gaug.train_predict_edges(data.adj, data.x, data.y, device, 30, args.removal_rate, args.add_rate)

            for epoch in range(args.epochs):
                model.initialize()
                if args.aug_type == 'flag_orig':
                    train_loss = train_flag_orig(data, model, optimizer, device, args)

                val_loss = validate(data, model)

                print(f'Run: {r + 1}, Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
                if lowest_val_loss > val_loss or epoch == args.epochs - 1:
                    lowest_val_loss = val_loss
                    evals = evaluate(model, data, device)
                    best_val = evals['val_f1']
                    best_test = evals['test_f1']
                    best_tr = evals['train_f1']
                early_stopping(val_loss, model)

                if early_stopping.early_stop or epoch == args.epochs - 1:
                    print(f'Train F1: {best_tr:.4f}, Validation F1: {best_val:.4f}, Test F1: {best_test:.4f}')
                    val_f1_list.append(best_val)
                    train_f1_list.append(best_tr)
                    test_f1_list.append(best_test)
                    break

            # file.write(f'{r + 1},{epoch},{best_tr:.4f},{best_val:.4f},{best_test:.4f}\n')

        file.write(f'total,{np.mean(train_f1_list):.4f}, {np.mean(val_f1_list):.4f},{np.mean(test_f1_list):.4f}\n')
        print(f'total,{np.mean(train_f1_list):.4f}, {np.mean(val_f1_list):.4f}, {np.mean(test_f1_list):.4f}\n')