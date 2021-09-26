from torch.nn import functional as F


def compute_consistency(pred, label):
    return F.kl_div(F.log_softmax(pred, dim=0), F.log_softmax(label, dim=0), reduction='mean').cpu()
