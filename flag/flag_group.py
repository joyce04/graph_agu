import torch

from grand.util import consis_loss

def apply_flag_group(model_forward, data, optimizer, device, args):
    train_idx = data.train_mask.nonzero().T[0]
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    pred_list = []
    out_list = []
    loss_list = []
    K = args.sample
    for i in range(K):
        perturb = torch.FloatTensor(*data.x.shape).uniform_(-args.step_size, args.step_size).to(device)
        perturb.requires_grad_()
        out = forward(perturb)
        pred_label = out.detach()
        pred_list.append(pred_label)
        loss = model.loss(out[train_idx],data.y[train_idx]) / args.m
        for _ in range(args.m - 1):
            loss.backward()
            perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            out = forward(perturb)
            loss = model.loss(out[train_idx],data.y[train_idx]) / args.m
        loss_list.append(loss)
        out_list.append(out)

    out = forward(perturb)
    loss = model.loss(out[train_idx], data.y[train_idx]) / args.m

    if args.cr:
        cr_loss = consis_loss(out_list, args.tem, args.lam)
        loss = loss + cr_loss

    loss.backward()
    optimizer.step()

    return loss