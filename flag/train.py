import torch

from cr.util import compute_consistency


def apply(model_forward, data, optimizer, device, args):
    m = args.m
    step_size = args.step_size
    cr_include = args.cr

    train_idx = data.train_mask.nonzero().T[0]
    perturb_shape = data.x.shape
    model, forward = model_forward

    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()

    out = forward(perturb)

    condition = torch.bitwise_or(data.ul_train_mask, data.train_mask)
    pred_label = out.detach()[condition]
    loss = model.loss(out[train_idx], data.y[train_idx]) / m

    for i in range(m - 1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = model.loss(out[train_idx], data.y[train_idx]) / m

    if cr_include:
        cr_loss = compute_consistency(pred_label, out[condition])
        loss = loss + cr_loss

    loss.backward()
    optimizer.step()

    return loss