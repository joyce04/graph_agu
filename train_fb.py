import random

import numpy as np
import torch
from torch.optim import Adam

from de.util import get_sampler
from eval import validate, evaluate
from flag.original import apply, apply_biased
from gaug.gaug import GAug
from gnn.clf import generate_node_clf
from util.config import get_arguments, device_setup
from util.data import dataset_split
from util.graph import csr_to_edgelist, lap_dinv
from util.tool import EarlyStopping


def train(data, model, optimizer, device, args, lap, d_inv):
    optimizer.zero_grad()
    out = model(data.x, data.train_index, lap, d_inv)
    loss = model.loss(out[data.train_mask == 1], data.y[data.train_mask == 1])
    loss.backward()
    optimizer.step()
    return loss


# def train_flag(data, model, optimizer, device, args):
#     forward = lambda perturb: model(data.x + perturb, data.train_index)
#     model_forward = (model, forward)
#     if args.biased:
#         loss = apply_biased(model_forward, data, optimizer, device, args)
#     else:
#         loss = apply(model_forward, data, optimizer, device, args)
#     return loss
#
#
# def train_de(data, model, optimizer, device, sampler, sampling_percent, normalization):
#     (train_adj, train_fea) = sampler.randomedge_sampler(percent=sampling_percent, normalization=normalization, cuda=(device == 'cuda'))
#     data.x = torch.Tensor(train_fea).to(device)
#     sampler.train_features = train_fea
#     edges = csr_to_edgelist(train_adj).type(torch.int64)
#
#     optimizer.zero_grad()
#
#     out = model(data.x, edges.to(device))
#     loss = model.loss(out[data.train_mask == 1], data.y[data.train_mask == 1])
#
#     loss.backward()
#     optimizer.step()
#
#     return loss
#
#
# def train_gaug(data, gaug, model, optimizer, device):
#     optimizer.zero_grad()
#
#     updated_edges = gaug.updated_edges
#     out = model(data.x, updated_edges)
#     loss = model.loss(out[data.train_mask == 1], data.y[data.train_mask == 1])
#
#     loss.backward()
#     optimizer.step()
#
#     return loss


if __name__ == '__main__':
    args = get_arguments()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_setup()
    if device == 'gpu':
        torch.cuda.manual_seed_all(args.seed)

    with open('./results/nc_{}_{}_{}_{}_es_{}.csv'.format(args.config.replace('.json', '').replace('./configs/', ''), args.gnn, args.epochs, args.dataset, str(args.edge_split)),
              'a+') as file:
        file.write(','.join(map(lambda x: x + ':' + str(vars(args)[x]), vars(args).keys())) + '\n')
        file.write('run, train F1 avg, validation F1 avg, test F1 avg\n')

        val_f1_list, test_f1_list, train_f1_list = [], [], []

        for r in range(10):
            dataset, data = dataset_split(args.data_loc, args.dataset, args.data_split, args.train_ratio, args.edge_split)
            num_nodes = data.x.shape[0]
            num_feats = data.x.shape[1]
            num_nd_classes = int(np.max(data.y.numpy()) + 1)

            if args.config.find('de.json') >= 0:
                sampler, data = get_sampler(data, data.adj, device)
                if args.gnn == 'gat':
                    dropout = 0.5
                else:
                    dropout = 0.8
            elif args.config.find('gaug.json') >= 0:
                if args.gaug_type == 'M':
                    gaug = GAug(True)
                    gaug.get_pretrained_edges(data, args.m_file_loc, args.removal_rate, args.add_rate)
                else:
                    gaug = GAug(False)
                    gaug.train_predict_edges(data.adj, data.x, data.y, device, 30, args.removal_rate, args.add_rate)

                dropout = 0.5
            else:
                dropout = 0.5

            if args.gnn == 'fbgcn' or args.gnn == 'fbgat':
                lap, d_inv = lap_dinv(data.edge_index, num_nodes)

            model = generate_node_clf(args.gnn, num_feats, num_nd_classes, dropout, device)
            model.reset_parameters()
            optimizer = Adam(model.gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            best_test, best_val, best_tr = 0, 0, 0
            lowest_val_loss = float("inf")

            model = model.to(device)
            data = data.to(device)
            lap = lap.to(device)
            d_inv = d_inv.to(device)
            for epoch in range(args.epochs):
                model.initialize()
                # if args.config.find('flag.json') >= 0:
                #     train_loss = train_flag(data, model, optimizer, device, args)
                # elif args.config.find('base.json') >= 0:
                train_loss = train(data, model, optimizer, device, args, lap, d_inv)
                # elif args.config.find('de.json') >= 0:
                #     train_loss = train_de(data, model, optimizer, device, sampler, args.de_sampling_percent, args.de_normalization)
                # elif args.config.find('gaug.json') >= 0:
                #     train_loss = train_gaug(data, gaug, model, optimizer, device)
                val_loss = validate(data, model, lap, d_inv)

                print(f'Run: {r + 1}, Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
                if lowest_val_loss > val_loss or epoch == args.epochs - 1:
                    lowest_val_loss = val_loss
                    evals = evaluate(model, data, device, lap, d_inv)
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

        file.write(f'total,{np.mean(train_f1_list):.4f}, {np.mean(val_f1_list):.4f}, {np.mean(test_f1_list):.4f}\n')
        print(f'total,{np.mean(train_f1_list):.4f}, {np.mean(val_f1_list):.4f}, {np.mean(test_f1_list):.4f}\n')
