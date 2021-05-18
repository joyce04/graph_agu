import logging
import random

import numpy as np
import optuna
import torch
from torch.optim import Adam

from de.util import get_sampler
from gaug.gaug import GAug
from gnn.clf import generate_node_clf
from util.config import get_arguments, device_setup
from util.data import dataset_split
from eval import validate, evaluate
from util.tool import EarlyStopping


def objective(trial):
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
        best_acc_test, best_acc_val, best_acc_tr = 0, 0, 0
        lowest_val_loss = float("inf")

        if args.config.find('de.json') >= 0:
            sampler, data = get_sampler(data, data.adj)
            sampling_percent = trial.suggest_float('de_sampling_percent', 0.1, 0.8)
            normalization = trial.suggest_categorical('de_normalization', ['FirstOrderGCN'])
            # ['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN', 'AugNormAdj', 'BingGeNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm', 'INorm']
        elif args.config.find('gaug.json') >= 0:
            removal_rate = trial.suggest_int('removal_rate', 10, 90)  # gaug_param['removal_rate']
            add_rate = trial.suggest_int('add_rate', 50, 150)  # gaug_param['add_rate']
            if args.gaug_type == 'M':
                gaug = GAug(True)
                gaug.get_pretrained_edges(data, args.m_file_loc, removal_rate, add_rate)
            else:
                gaug = GAug(False)
                gaug.train_predict_edges(data.adj, data.x, data.y, device, 30, removal_rate, add_rate)
        else:
            raise Exception('train function not defined')

        for epoch in range(args.epochs):
            model.initialize()

            if args.config.find('de.json') >= 0:
                train_loss = train(data, model, optimizer, device, sampler, sampling_percent, normalization)
            elif args.config.find('gaug.json') >= 0:
                train_loss = train(data, gaug, model, optimizer, device)
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

    return np.mean(val_acc_list)


if __name__ == '__main__':
    args = get_arguments()
    print(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_setup()

    logging.info('Optimization with Optuna')
    study = optuna.create_study(direction='maximize')

    if args.config.find('de.json') >= 0:
        from train_base import train_de as train
    elif args.config.find('gaug.json') >= 0:
        from train_base import train_gaug as train

    study.optimize(objective, n_trials=10)

    with open('./results/nc_optuna_{}_{}_{}_{}_es_{}.txt'.format(args.config.replace('.json', '').replace('./configs/', ''),
                                                                 args.gnn, args.epochs, args.dataset, str(args.edge_split)), 'a+') as file:
        file.write(','.join(map(lambda x: x + ':' + str(vars(args)[x]), vars(args).keys())) + '\n')
        file.write('run, epoch, train F1 avg, train acc avg, validation F1 avg,validation acc avg, test F1 avg, test acc avg\n')

        file.write('Number of finished trials: ' + str(len(study.trials)) + '\n')
        trial = study.best_trial
        file.write('Best trial\n')
        file.write('Value: ' + str(trial.value) + '\n')
        file.write('Params\n')
        for key, value in trial.params.items():
            file.write('{}: {}\n'.format(key, value))
