# -*- coding: utf-8 -*-
from test import test  
from config.config import (
    DEVICE, MODEL_SAVE_PATH, class_names, batch_size
)
from data.data_loader import dataloader
import torch

def test_main():

    device = DEVICE
    class_labels = class_names


    model = torch.load(MODEL_SAVE_PATH, map_location=device)
    model = model.to(device)  


    _, _, test_loader = dataloader(batch_size=batch_size)


    test.evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_labels,
        cm_path="results/confusion_matrix.png",
        metrics_path="results/metrics.txt"
    )
