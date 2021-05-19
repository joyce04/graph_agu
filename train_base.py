import random

import numpy as np
import torch
from torch.optim import Adam

from eval import validate, evaluate
from gnn.clf import generate_node_clf
from util.config import get_arguments, device_setup
from util.data import dataset_split
from util.tool import EarlyStopping
from flag.original import apply
from util.graph import csr_to_edgelist
from de.util import get_sampler
from gaug.gaug import GAug


def train(data, model, optimizer, device, args):
    optimizer.zero_grad()

    out = model(data.x, data.train_index)
    loss = model.loss(out[data.train_mask == 1], data.y[data.train_mask == 1])
    loss.backward()
    optimizer.step()
    return loss


def train_flag(data, model, optimizer, device, args):
    forward = lambda perturb: model(data.x + perturb, data.train_index)
    model_forward = (model, forward)
    loss = apply(model_forward, data, optimizer, device, args)
    return loss


def train_de(data, model, optimizer, device, sampler, sampling_percent, normalization):
    (train_adj, train_fea) = sampler.randomedge_sampler(percent=sampling_percent, normalization=normalization, cuda=(device == 'cuda'))
    data.x = torch.Tensor(train_fea).to(device)
    sampler.train_features = train_fea
    edges = csr_to_edgelist(train_adj).type(torch.int64)

    optimizer.zero_grad()

    out = model(data.x, edges.to(device))
    loss = model.loss(out[data.train_mask == 1], data.y[data.train_mask == 1])

    loss.backward()
    optimizer.step()

    return loss


def train_gaug(data, gaug, model, optimizer, device):
    optimizer.zero_grad()

    updated_edges = gaug.updated_edges
    out = model(data.x, updated_edges)
    loss = model.loss(out[data.train_mask == 1], data.y[data.train_mask == 1])

    loss.backward()
    optimizer.step()

    return loss


if __name__ == '__main__':
    args = get_arguments()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_setup()

    with open('./results/nc_{}_{}_{}_{}_es_{}.csv'.format(args.config.replace('.json', '').replace('./configs/', ''), args.gnn, args.epochs, args.dataset, str(args.edge_split)),
              'a+') as file:
        file.write(','.join(map(lambda x: x + ':' + str(vars(args)[x]), vars(args).keys())) + '\n')
        file.write('run, train F1 avg, train acc avg, validation F1 avg,validation acc avg, test F1 avg, test acc avg\n')

        val_f1_list, test_f1_list, train_f1_list = [], [], []
        val_acc_list, test_acc_list, train_acc_list = [], [], []

        for r in range(10):
            dataset, data = dataset_split(args.data_loc, args.dataset, args.data_split, args.train_ratio, args.edge_split)
            num_nodes = data.x.shape[0]
            num_feats = data.x.shape[1]
            num_nd_classes = np.max(data.y.numpy()) + 1

            model = generate_node_clf(args.gnn, num_feats, num_nd_classes, device)
            optimizer = Adam(model.gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            best_test, best_val, best_tr = 0, 0, 0
            best_acc_test, best_acc_val, best_acc_tr = 0, 0, 0
            lowest_val_loss = float("inf")

            if args.config.find('de.json') >= 0:
                sampler, data = get_sampler(data, data.adj, device)
            elif args.config.find('gaug.json') >= 0:
                if args.gaug_type == 'M':
                    gaug = GAug(True)
                    gaug.get_pretrained_edges(data, args.m_file_loc, args.removal_rate, args.add_rate)
                else:
                    gaug = GAug(False)
                    gaug.train_predict_edges(data.adj, data.x, data.y, device, 30, args.removal_rate, args.add_rate)

            data.x = data.x.to(device)
            data.train_index = data.train_index.to(device)
            data.y = data.y.to(device)
            for epoch in range(args.epochs):
                model.initialize()
                if args.config.find('flag.json') >= 0:
                    train_loss = train_flag(data, model, optimizer, device, args)
                elif args.config.find('base.json') >= 0:
                    train_loss = train(data, model, optimizer, device, args)
                elif args.config.find('de.json') >= 0:
                    train_loss = train_de(data, model, optimizer, device, sampler, args.de_sampling_percent, args.de_normalization)
                elif args.config.find('gaug.json') >= 0:
                    train_loss = train_gaug(data, gaug, model, optimizer, device)
                val_loss = validate(data, model)

                print(f'Run: {r + 1}, Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
                if lowest_val_loss > val_loss or epoch == args.epochs - 1:
                    lowest_val_loss = val_loss
                    evals = evaluate(model, data, device)
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
