# -*- coding: utf-8 -*-
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')  # 支持中文显示

def model_train(batch_size, epochs, train_loader, val_loader, model, optimizer, loss_function, device, model_save_path,loss_save_path):
    model = model.to(device)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    best_accuracy = 0.0
    best_model = model

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)

            probabilities = F.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            epoch_correct += (predicted == y).sum().item()

            loss = loss_function(output, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_accuracy = epoch_correct / train_size
        train_loss_history.append(epoch_loss / train_size)
        train_acc_history.append(epoch_accuracy)
        print(f'Epoch {epoch+1:3} | Train Loss: {epoch_loss/train_size:.6f} | Train Acc: {epoch_accuracy:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                probabilities = F.softmax(output, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                val_correct += (predicted == y).sum().item()
                loss = loss_function(output, y)
                val_loss += loss.item()

        val_accuracy = val_correct / val_size
        val_loss_history.append(val_loss / val_size)
        val_acc_history.append(val_accuracy)
        print(f'Epoch {epoch+1:3} | Val Loss: {val_loss/val_size:.6f} | Val Acc: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            torch.save(best_model, model_save_path)
            print(">>> Best model saved.")

    duration = time.time() - start_time
    print(f'\nTraining complete in {duration:.2f} seconds. Best Val Acc: {best_accuracy:.4f}')


    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_loss_history, label='Val Loss')
    plt.plot(val_acc_history, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    

    plt.savefig(loss_save_path)  
    plt.close()  

