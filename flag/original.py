import torch

from cr.util import compute_consistency


def apply(model_forward, data, optimizer, device, args):
    train_idx = data.train_mask.nonzero().T[0]
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*data.x.shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    condition = data.ul_train_mask == 1
    pred_label = out.detach()[condition]
    loss = model.loss(out[train_idx], data.y[train_idx]) / args.m

    for _ in range(args.m - 1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = model.loss(out[train_idx], data.y[train_idx]) / args.m

    if args.cr and args.data_split != 'full':
        cr_loss = compute_consistency(pred_label, out[condition])
        loss = loss + cr_loss

    loss.backward()
    optimizer.step()

    return loss
