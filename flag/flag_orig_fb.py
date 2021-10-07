import torch

from cr.util import compute_consistency


def apply_orig_flag_gaug(model_forward, data, gaug, optimizer, device, args):
    # THRESHOLD = aug_params['THRESHOLD']
    m = args.m
    step_size = args.step_size
    model, forward = model_forward
    train_idx = data.train_mask.nonzero().T[0]

    # condition = data.ul_train_mask == 1

    model.train()
    optimizer.zero_grad()

    node_cls = model(data.x, data.train_index)
    loss = model.loss(node_cls[train_idx], data.y[train_idx]) / m

    pred_label = node_cls.detach()
    # pred_label = node_cls[condition].detach()
    # over_threshold = F.softmax(pred_label).max(dim=1)[0] > THRESHOLD
    # no_pred_over_threshold = pred_label[over_threshold].shape[0] == 0
    # if no_pred_over_threshold:
    #     print('pseudo label accuracy with THRESHOLD :  0')
    # else:
    #     print('pseudo label accuracy with THRESHOLD : ',
    #           data.y[condition][over_threshold].eq(pred_label[over_threshold].max(dim=1)[1]).sum().item() / data.y[condition][over_threshold].shape[0])

    perturb = torch.FloatTensor(*data.x.shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()

    for i in range(m - 1):
        loss.backward()
        if i == m - 2:
            data.gaug = gaug.updated_edges
            node_cls = forward(perturb, data.gaug)
        else:
            node_cls = forward(perturb, data.train_index)

        loss = model.loss(node_cls[train_idx], data.y[train_idx]) / m

        if 0 < i < m - 1:
            perturb.data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.grad[:] = 0

    if args.cr:
        # if not args['threshold_CR'] and no_pred_over_threshold:
        #     cr_loss = 0
        # else:
        cr_loss = compute_consistency(node_cls, pred_label)
        loss = loss + cr_loss

    loss.backward()
    optimizer.step()

    return loss


def apply_flag_orig(model_forward, data, optimizer, device, args,lsym):
    # THRESHOLD = params['THRESHOLD']
    m = args.m
    step_size = args.step_size

    train_idx = data.train_mask.nonzero().T[0]
    model, forward = model_forward
    model = model.to(device)
    forward = forward
    model.train()
    optimizer.zero_grad()

    node_cls = model(data.x, data.train_index,lsym)

    # condition = data.ul_train_mask == 1
    loss = model.loss(node_cls[train_idx], data.y[train_idx]) / (m + 1)
    pred_label = node_cls.detach()

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

    if args.cr:
        # if not aug_params['threshold_CR'] and no_pred_over_threshold:
        #     cr_loss = 0
        # else:
        cr_loss = compute_consistency(node_cls, pred_label)
        loss = loss + cr_loss

    loss.backward()
    optimizer.step()

    return loss
