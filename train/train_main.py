# -*- coding: utf-8 -*-
import torch
from torch import optim, nn
from config.config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH,LOSS_SAVE_PATH, batch_size, input_dim,hidden_layer_sizes,output_dim, attention_dim
from models.lstm import LSTMAttentionClassifier
from data.data_loader import dataloader
from train.train import model_train

def train_main():
    # Load data
    train_loader, val_loader, _ = dataloader(batch_size=batch_size)

    # Initialize model
    model = LSTMAttentionClassifier(input_dim,hidden_layer_sizes,output_dim, attention_dim)

    # Loss and optimizer
    loss_function = nn.CrossEntropyLoss(reduction='sum')
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
