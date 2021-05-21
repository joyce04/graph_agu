import torch

from cr.util import compute_consistency


def apply_flag_orig(model_forward, data, optimizer, device, args):
    # THRESHOLD = params['THRESHOLD']
    m = args.m
    step_size = args.step_size

    train_idx = data.train_mask.nonzero().T[0]
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    node_cls = model(data.x, data.train_index)

    condition = data.ul_train_mask == 1
    loss = model.loss(node_cls[train_idx], data.y[train_idx]) / (m + 1)
    pred_label = node_cls[condition].detach()

    # over_threshold = F.softmax(pred_label).max(dim=1)[0] > THRESHOLD
    # no_pred_over_threshold = pred_label[over_threshold].shape[0] == 0
    # if no_pred_over_threshold:
    #     print('pseudo label accuracy with THRESHOLD :  0')
    # else:
    #     print('pseudo label accuracy with THRESHOLD : ',
    #           data.y[condition][over_threshold].eq(pred_label[over_threshold].max(dim=1)[1]).sum().item() / data.y[condition][over_threshold].shape[0])

    loss.backward()

    perturb = torch.FloatTensor(*data.x.shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()

    node_cls = forward(perturb)
    loss = model.loss(node_cls[train_idx], data.y[train_idx]) / (m + 1)

    for i in range(m - 1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        node_cls = forward(perturb)
        loss = model.loss(node_cls[train_idx], data.y[train_idx]) / (m + 1)

    if args.cr and args.data_split != 'full':
        # if not aug_params['threshold_CR'] and no_pred_over_threshold:
        #     cr_loss = 0
        # else:
        cr_loss = compute_consistency(node_cls[condition], pred_label)
        loss = loss + cr_loss

    loss.backward()
    optimizer.step()

    return loss