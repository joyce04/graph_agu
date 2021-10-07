import random

import numpy as np
import torch
from torch import optim
from torch.optim import Adam

# from de.util import get_sampler
from eval import validate_fb, evaluate_fb
from flag.original import apply, apply_biased
# from gaug.gaug import GAug
from gnn.clf import generate_node_clf
from util.config import get_arguments, device_setup
from util.data import dataset_split
from util.tool import EarlyStopping
from grand.util import adj_for_rand_propagate, rand_prop, consis_loss

def train(data, model, optimizer, device, args, lsym):
    optimizer.zero_grad()
    out = model(data.x, data.train_index, lsym)
    loss = model.loss(out[data.train_mask == 1], data.y[data.train_mask == 1])
    loss.backward()
    optimizer.step()
    return loss


def train_flag(data, model, optimizer, device, args, lsym):
    forward = lambda perturb: model(data.x + perturb, data.train_index, lsym)
    model_forward = (model, forward)
    if args.biased:
        loss = apply_biased(model_forward, data, optimizer, device, args)
    else:
        loss = apply(model_forward, data, optimizer, device, args)
    return loss




def train_grand(data, model, optimizer, device, lsym, K, tem, lam, order):
    optimizer.zero_grad()
    x = data.x
    A = adj_for_rand_propagate(data.adj)
    x_list = []
    out_list = []
    loss = 0
    for k in range(K):
        x_list.append(rand_prop(x, order, A))
        out_list.append(torch.log_softmax(model(x_list[k], data.train_index, lsym), dim = -1))
        loss += model.loss(out_list[k][data.train_mask == 1], data.y[data.train_mask == 1])
    loss = loss/K
    loss_consis = consis_loss(out_list, temp = tem, lam = lam)
    loss = loss + loss_consis
    
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
    if device == 'gpu':
        torch.cuda.manual_seed_all(args.seed)

    with open('./results/nc_{}_{}_{}_{}_es_{}.csv'.format(args.config.replace('.json', '').replace('./configs/', ''), args.gnn, args.epochs, args.dataset, str(args.edge_split)),
              'a+') as file:
        file.write(','.join(map(lambda x: x + ':' + str(vars(args)[x]), vars(args).keys())) + '\n')
        file.write('run, train F1 avg, validation F1 avg, test F1 avg\n')

        val_f1_list, test_f1_list, train_f1_list = [], [], []

        for r in range(3):
            dataset, data = dataset_split(args.data_loc, args.dataset, args.data_split, args.train_ratio, args.edge_split)
            num_nodes = data.x.shape[0]
            num_feats = data.x.shape[1]
            num_nd_classes = int(np.max(data.y.numpy()) + 1)

            dropout = 0.5
            model = generate_node_clf(args.gnn, num_feats, num_nd_classes, dropout, device, data.aL, data.aH)
            model.reset_parameters()
            optimizer = Adam(model.gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            best_test, best_val, best_tr = 0, 0, 0
            lowest_val_loss = float("inf")
            model = model.to(device)
            data = data.to(device)
            
            for epoch in range(args.epochs):
                model.initialize()
                if args.config.find('flag.json') >= 0:
                    train_loss = train_flag(data, model, optimizer, device, args, data.lsym)
                if args.config.find('base.json') >= 0:
                    train_loss = train(data, model, optimizer, device, args,  data.lsym)
                elif args.config.find('grand.json') >= 0:
                    train_loss = train_grand(data, model, optimizer, device, data.lsym, args.sample, args.tem, args.lam, args.order)
                val_loss = validate_fb(data, model, data.lsym)

                print(f'Run: {r + 1}, Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
                if lowest_val_loss > val_loss or epoch == args.epochs - 1:
                    lowest_val_loss = val_loss
                    evals = evaluate_fb(model, data, device, data.lsym)
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
        for name, param in model.named_parameters():
            if name in ["gnn_model.fbgcns.0.aL", "gnn_model.fbgcns.0.aH", "gnn_model.fbgcns.1.aL", "gnn_model.fbgcns.1.aH"]:
                print (name, param.data)