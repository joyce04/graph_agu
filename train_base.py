import argparse

import numpy as np
import torch
import random

from torch.optim import Adam

from eval import validate, evaluate, evaluate_sage, validate_sage
from gnn.clf import generate_node_clf
from util.data import dataset_split
from util.tool import EarlyStopping
from torch_geometric.data import NeighborSampler


def train(data, model, optimizer):
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
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
    parser = argparse.ArgumentParser()
    # Training parameter
    parser.add_argument('--file_loc', default='../dataset/', help='dataset file location')
    parser.add_argument('--dataset', default='cora', help='benchmark dataset : cora, citeseer')
    parser.add_argument('--data_split', default='public', help='public, full')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--gnn', default='gat', help='gcn, graphsage, gat')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--learning_rate', default=0.003, help='learning rate')
    parser.add_argument('--weight_decay', default=0.0005, help='weight decay')
    parser.add_argument('--device', default='cpu', help='cpu or gpu{num}')
    args = parser.parse_args()

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    device = args.device
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dataset = dataset_split(args.file_loc, args.dataset, split_type=args.data_split)
    data = dataset[0]
    num_nodes = data.x.shape[0]
    num_feats = data.x.shape[1]
    num_nd_classes = np.max(data.y.numpy()) + 1

    if args.gnn == 'graphsage':
        BATCH_SIZE = 20
        train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask, num_nodes=num_nodes,
                                       sizes=[25, 10], batch_size=BATCH_SIZE, shuffle=True)
        # full graph for evaluation
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                          batch_size=BATCH_SIZE, shuffle=False)

    data.x = data.x.to(device)
    model = generate_node_clf(args.gnn, num_feats, num_nd_classes, device)
    optimizer = Adam(model.gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=100, verbose=True)

    best_test, best_val, best_tr = 0, 0, 0
    lowest_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.initialize()
        if args.gnn == 'graphsage':
            train_loss = train_sage(train_loader, model, optimizer)
            val_loss = validate_sage(data, model, subgraph_loader, device)
        else:
            train_loss = train(data, model, optimizer)
            val_loss = validate(data, model)

        print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
        if lowest_val_loss > val_loss or epoch == args.epochs - 1:
            lowest_val_loss = val_loss

            if args.gnn == 'graphsage':
                evals = evaluate_sage(model, data, subgraph_loader, device)
            else:
                evals = evaluate(model, data)
            best_val = evals['val_f1']
            best_test = evals['test_f1']
            best_tr = evals['train_f1']

        early_stopping(val_loss, model)
        if early_stopping.early_stop or epoch == args.epochs - 1:
            print(f'Train F1: {best_tr:.4f}, Validation F1: {best_val:.4f}, Test F1: {best_test:.4f}')
            break

    with open('./results/nc_base_{}_{}_{}.csv'.format(args.gnn, args.epochs, args.dataset), 'w') as file:
        file.write('epoch, train F1 avg, validation F1 avg, test F1 avg\n')
        file.write(f'{epoch},{best_tr:.4f},{best_val:.4f},{best_test:.4f}\n')
