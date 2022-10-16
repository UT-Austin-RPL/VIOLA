from torch import nn

class LossFn(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.loss_fn = None

    def forward(self, prediciton, target):
        raise NotImplementedError

    def valid_loss(self, prediction, target):
        # maybe want to use it in the future
        raise NotImplementedError

class L2Loss(nn.Module):
    def __init__(self, reduction="sum"):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="sum")

    def forward(self, prediction, target):
        return self.loss_fn(prediction, target)

class NLLLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

    def forward(self, prediction, target):
        # Make sure prediction is in the form of distributions
        log_probs = prediction.log_prob(target)
        loss = -log_probs
        return loss.mean()

