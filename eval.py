import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score


@torch.no_grad()
def validate(data, model):
    model.evaluate()

    out = model(data.x, data.edge_index)
    return model.loss(out[data.val_mask == 1], data.y[data.val_mask == 1])


@torch.no_grad()
def validate_sage(data, model, subgraph_loader, device):
    model.evaluate()

    out = model.gnn_model.inference(data.x, subgraph_loader, device)
    return model.loss(out[data.val_mask == 1], data.y[data.val_mask == 1])


def evaluate_f1(data, out):
    outputs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.cross_entropy(out[mask == 1], data.y[mask == 1]).item()
        pred = out[mask == 1].max(dim=1)[1]

        outputs['{}_loss'.format(key)] = loss
        outputs['{}_f1'.format(key)] = f1_score(data.y[mask == 1], pred.data.numpy(), average='micro')
    return outputs


@torch.no_grad()
def evaluate(model, data):
    model.evaluate()
    out = model(data.x, data.edge_index)

    return evaluate_f1(data, out)


@torch.no_grad()
def evaluate_sage(model, data, subgraph_loader, device):
    model.evaluate()
    out = model.gnn_model.inference(data.x, subgraph_loader, device)

    return evaluate_f1(data, out)
