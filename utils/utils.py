# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(cm, labels, save_path, annot=True, fmt='d', cmap='Blues'):

    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, save_path):
    try:
        with open(save_path, 'w') as f:

            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"accuracy: {metrics}\n")  
    except Exception as e:
        print(f"保存时候出错: {e}")