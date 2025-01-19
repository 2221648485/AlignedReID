import torch


def init_optim(optim, params, lr, weight_decay):
    if optim == "adam":
        return torch.optim.Adam(params, lr, weight_decay=weight_decay)
    elif optim == "sgd":
        return torch.optim.SGD(params, lr, weight_decay=weight_decay)
    elif optim == "rmsprop":
        return torch.optim.RMSprop(params, lr, weight_decay=weight_decay)
    else:
        raise KeyError(f"Unsupported optim: {optim}")
