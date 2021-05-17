import random

import numpy as np
import torch
from torch.optim import Adam

from eval import validate, evaluate
from flag.train import apply
from gnn.clf import generate_node_clf
from util.config import get_arguments, device_setup
from util.data import dataset_split
from util.tool import EarlyStopping


def train(data, model, optimizer, device, args):
    forward = lambda perturb: model(data.x + perturb, data.train_index)
    model_forward = (model, forward)
    loss = apply(model_forward, data, optimizer, device, args)
    return loss


if __name__ == '__main__':
    args = get_arguments()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_setup()

    with open('./results/nc_flag_{}_{}_{}_es_{}_cr{}.csv'.format(args.gnn, args.epochs, args.dataset, str(args.edge_split), str(args.cr)), 'a+') as file:
        file.write(','.join(map(lambda x: x + ':' + str(vars(args)[x]), vars(args).keys())) + '\n')
        file.write('run, epoch, train F1 avg, validation F1 avg, test F1 avg\n')

        val_f1_list, test_f1_list, train_f1_list = [], [], []
        val_acc_list, test_acc_list, train_acc_list = [], [], []

        for r in range(10):
            dataset, data = dataset_split(args.data_loc, args.dataset, args.data_split, args.train_ratio, args.edge_split)
            num_feats = data.x.shape[1]
            num_nd_classes = np.max(data.y.numpy()) + 1

            data.x = data.x.to(device)
            model = generate_node_clf(args.gnn, num_feats, num_nd_classes, device)
            optimizer = Adam(model.gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            best_test, best_val, best_tr = 0, 0, 0
            lowest_val_loss = float("inf")

            for epoch in range(args.epochs):
                model.initialize()

                train_loss = train(data, model, optimizer, device, args)
                val_loss = validate(data, model)

                print(f'Run: {r + 1}, Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
                if lowest_val_loss > val_loss or epoch == args.epochs - 1:
                    lowest_val_loss = val_loss
                    evals = evaluate(model, data)
                    best_val = evals['val_f1']
                    best_test = evals['test_f1']
                    best_tr = evals['train_f1']
                    best_acc_val = evals['val_acc']
                    best_acc_test = evals['test_acc']
                    best_acc_tr = evals['train_acc']
                early_stopping(val_loss, model)

                if early_stopping.early_stop or epoch == args.epochs - 1:
                    print(f'Train F1: {best_tr:.4f}, Validation F1: {best_val:.4f}, Test F1: {best_test:.4f}')
                    val_f1_list.append(best_val)
                    train_f1_list.append(best_tr)
                    test_f1_list.append(best_test)
                    val_acc_list.append(best_acc_val)
                    train_acc_list.append(best_acc_tr)
                    test_acc_list.append(best_acc_test)
                    break

            # file.write(f'{r + 1},{epoch},{best_tr:.4f},{best_val:.4f},{best_test:.4f}\n')

        file.write(
            f'total,{np.mean(train_f1_list):.4f}, {np.mean(train_acc_list):.4f},{np.mean(val_f1_list):.4f}, {np.mean(val_acc_list):.4f},{np.mean(test_f1_list):.4f}, {np.mean(test_acc_list):.4f}\n')
        print(
            f'total,{np.mean(train_f1_list):.4f}, {np.mean(train_acc_list):.4f},{np.mean(val_f1_list):.4f}, {np.mean(val_acc_list):.4f},{np.mean(test_f1_list):.4f}, {np.mean(test_acc_list):.4f}\n')
