import torch
import torch.nn as nn


def loss_function(output_samples_classification, target, output_samples_score, device, args):
    
    class_loss = nn.CrossEntropyLoss()
    cl = class_loss(output_samples_classification, target[0])
    rl = None

    if args.inos:
        inos_loss = getattr(nn, args.inos_loss)
        # nn.BCEWithLogitsLoss()
        rl = inos_loss(output_samples_score,target[1])

    return cl, rl
