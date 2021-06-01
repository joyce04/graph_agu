import torch

from cr.util import compute_consistency

'''
Code adopted from https://github.com/devnkong/FLAG/blob/7e48d9194d3a7f515335cc351a663d65e09c2e1c/deep_gcns_torch/examples/ogb/attacks.py
'''


def apply(model_forward, data, optimizer, device, args):
    train_idx = data.train_mask.nonzero().T[0]
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*data.x.shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    pred_label = out.detach()
    loss = model.loss(out[train_idx], data.y[train_idx]) / args.m

    for _ in range(args.m - 1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = model.loss(out[train_idx], data.y[train_idx]) / args.m

    if args.cr:
        cr_loss = compute_consistency(pred_label, out)
        loss = loss + cr_loss

    loss.backward()
    optimizer.step()

    return loss


def apply_biased(model_forward, data, optimizer, device, args):
    train_idx = data.train_mask.nonzero().T[0]
    unlabel_idx = data.train_mask == 1  # list(set(range(data.x.shape[0])) - set(training_idx))

    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*data.x.shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.data[unlabel_idx] *= 2
    perturb.requires_grad_()

    out = forward(perturb)
    pred_label = out.detach()
    loss = model.loss(out[train_idx], data.y[train_idx]) / args.m

    for _ in range(args.m - 1):
        loss.backward()

        perturb_data_training = perturb[train_idx].detach() + args.step_size * torch.sign(perturb.grad[train_idx].detach())
        perturb.data[train_idx] = perturb_data_training.data

        perturb_data_unlabel = perturb[unlabel_idx].detach() + 2 * args.step_size * torch.sign(perturb.grad[unlabel_idx].detach())
        perturb.data[unlabel_idx] = perturb_data_unlabel.data

        perturb.grad[:] = 0
        out = forward(perturb)
        loss = model.loss(out[train_idx], data.y[train_idx]) / args.m

    if args.cr:
        cr_loss = compute_consistency(pred_label, out)
        loss = loss + cr_loss

    loss.backward()
    optimizer.step()

    return loss
