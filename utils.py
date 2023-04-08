from torchmetrics.classification import MulticlassF1Score
import torch


def metrics(preds, target):
    metr = MulticlassF1Score(num_classes=3).to('cpu')

    return metr(preds, target)
