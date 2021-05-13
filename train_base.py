import random

import numpy as np
import torch
from torch.optim import Adam

from eval import validate, evaluate, evaluate_sage, validate_sage
from gnn.clf import generate_node_clf
from gnn.sage import generate_sage_loader
from util.config import get_arguments, device_setup
from util.data import dataset_split
from util.tool import EarlyStopping


def train(data, model, optimizer):
    optimizer.zero_grad()

    out = model(data.x, data.train_index)
    loss = model.loss(out[data.train_mask == 1], data.y[data.train_mask == 1])
    loss.backward()
    optimizer.step()
    return loss


def train_sage(train_loader, model, optimizer):
    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        optimizer.zero_grad()

        adjs = [adj.to(device) for adj in adjs]
        out = model(data.x[n_id], adjs)
        labels = data.y[n_id][:batch_size]
        if labels.size(0) == 1:
            loss = model.loss(out, labels)
        else:
            loss = model.loss(out, labels.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(labels).sum())

    loss = total_loss / len(train_loader)
    return loss


if __name__ == '__main__':
    args = get_arguments()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_setup()

    with open('./results/nc_base_{}_{}_{}_es_{}.csv'.format(args.gnn, args.epochs, args.dataset, str(args.edge_split)), 'a+') as file:
        file.write(','.join(map(lambda x: x + ':' + str(vars(args)[x]), vars(args).keys())) + '\n')
        file.write('run, epoch, train F1 avg, train acc avg, validation F1 avg,validation acc avg, test F1 avg, test acc avg\n')

        val_f1_list, test_f1_list, train_f1_list = [], [], []
        val_acc_list, test_acc_list, train_acc_list = [], [], []

        for r in range(10):
            dataset, data = dataset_split(args.data_loc, args.dataset, args.data_split, args.train_ratio, args.edge_split)
            num_nodes = data.x.shape[0]
            num_feats = data.x.shape[1]
            num_nd_classes = np.max(data.y.numpy()) + 1

            if args.gnn == 'graphsage':
                BATCH_SIZE = 20
                train_loader, subgraph_loader = generate_sage_loader(data.train_index, data.train_mask, num_nodes, BATCH_SIZE)

            data.x = data.x.to(device)
            model = generate_node_clf(args.gnn, num_feats, num_nd_classes, device)
            optimizer = Adam(model.gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            best_test, best_val, best_tr = 0, 0, 0
            best_acc_test, best_acc_val, best_acc_tr = 0, 0, 0
            lowest_val_loss = float("inf")

            for epoch in range(args.epochs):
                model.initialize()
                if args.gnn == 'graphsage':
                    train_loss = train_sage(train_loader, model, optimizer)
                    val_loss = validate_sage(data, model, subgraph_loader, device)
                else:
                    train_loss = train(data, model, optimizer)
                    val_loss = validate(data, model)

                print(f'Run: {r + 1}, Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
                if lowest_val_loss > val_loss or epoch == args.epochs - 1:
                    lowest_val_loss = val_loss
                    if args.gnn == 'graphsage':
                        evals = evaluate_sage(model, data, subgraph_loader, device)
                    else:
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
