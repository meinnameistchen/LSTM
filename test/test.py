# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import plot_confusion_matrix, save_metrics

def evaluate_model(model, test_loader, device, class_names, cm_path, metrics_path):
    model.eval()
    model = model.to(device)
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            preds = torch.argmax(probabilities, dim=1)

            true_labels.extend(label.cpu().tolist())
            predicted_labels.extend(preds.cpu().tolist())


    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4, output_dict=True)
    plot_confusion_matrix(cm, class_names, cm_path)
    save_metrics(report, metrics_path)  
    return cm, report
