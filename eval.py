import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score


@torch.no_grad()
def validate(data, model):
    model.evaluate()

    out = model(data.x, data.train_index)
    return model.loss(out[data.val_mask == 1], data.y[data.val_mask == 1])


@torch.no_grad()
def validate_fb(data, model, lap, d_inv):
    model.evaluate()

    out = model(data.x, data.train_index, lap, d_inv)
    return model.loss(out[data.val_mask == 1], data.y[data.val_mask == 1])


# @torch.no_grad()
# def validate_sage(data, model, subgraph_loader, device):
#     model.evaluate()
#
#     out = model.gnn_model.inference(data.x, subgraph_loader, device)
#     return model.loss(out[data.val_mask == 1], data.y[data.val_mask == 1])


def evaluate_metrics(data, out, device):
    outputs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.cross_entropy(out[mask == 1], data.y[mask == 1]).item()
        pred = out[mask == 1].max(dim=1)[1]

        outputs['{}_loss'.format(key)] = loss
        if device == 'cpu':
            outputs['{}_f1'.format(key)] = f1_score(data.y[mask == 1], pred.data.numpy(), average='micro')
            outputs['{}_acc'.format(key)] = accuracy_score(data.y[mask == 1], pred.data.numpy())
        else:
            outputs['{}_f1'.format(key)] = f1_score(data.y[mask == 1].cpu(), pred.data.cpu().numpy(), average='micro')
            outputs['{}_acc'.format(key)] = accuracy_score(data.y[mask == 1].cpu(), pred.data.cpu().numpy())
    return outputs


@torch.no_grad()
def evaluate(model, data, device):
    model.evaluate()
    out = model(data.x, data.train_index)

    return evaluate_metrics(data, out, device)


@torch.no_grad()
def evaluate_fb(model, data, device, lap, d_inv):
    model.evaluate()
    out = model(data.x, data.train_index, lap, d_inv)

    return evaluate_metrics(data, out, device)

# @torch.no_grad()
# def evaluate_sage(model, data, subgraph_loader, device):
#     model.evaluate()
#     out = model.gnn_model.inference(data.x, subgraph_loader, device)
#
#     return evaluate_metrics(data, out)
