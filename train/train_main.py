# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import optim, nn
from config.config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH,LOSS_SAVE_PATH, batch_size, input_dim,hidden_layer_sizes,output_dim, attention_dim
from models.lstm import LSTMAttentionClassifier
from data.data_loader import dataloader
from train.train import model_train

class DynamicWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, dynamic=True, smooth_eps=1e-5):

        super().__init__()
        self.num_classes = num_classes
        self.dynamic = dynamic
        self.smooth_eps = smooth_eps
        self.register_buffer("default_weights", torch.ones(num_classes))

    def forward(self, inputs, targets):
        if self.dynamic:
            class_counts = torch.bincount(targets, minlength=self.num_classes).float()
            class_weights = 1.0 / (class_counts + self.smooth_eps)
            class_weights = class_weights / class_weights.sum() * self.num_classes
        else:
            class_weights = self.default_weights.to(inputs.device)

        sample_weights = class_weights[targets]
        loss = F.cross_entropy(inputs, targets, reduction='none')
        weighted_loss = loss * sample_weights
        return weighted_loss.mean()


def train_main():
    # Load data
    train_loader, val_loader, _ = dataloader(batch_size=batch_size)

    # Initialize model
    model = LSTMAttentionClassifier(input_dim,hidden_layer_sizes,output_dim, attention_dim)

    # Loss and optimizer
    loss_function = DynamicWeightedCrossEntropyLoss(output_dim,dynamic=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training
    model_train(
        batch_size=batch_size,
        epochs=EPOCHS,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=DEVICE,
        model_save_path=MODEL_SAVE_PATH,
        loss_save_path=LOSS_SAVE_PATH
    )
