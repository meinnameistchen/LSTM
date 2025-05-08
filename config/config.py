# -*- coding: utf-8 -*-
import torch
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型配置
MODEL_SAVE_PATH = "saved_models/best_model.pt"
LOSS_SAVE_PATH = 'results/loss_result.png'
class_names_dict = {
    0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5",
    5: "C6", 6: "C7", 7: "C8", 8: "C9", 9: "C10"
}
# 提取为 list（按照键顺序）
class_names = [class_names_dict[i] for i in range(len(class_names_dict))]
# 超参数
EPOCHS = 30
LEARNING_RATE = 0.001
# Configuration file for CNN1DModel
batch_size = 32
input_dim = 32   # 输入维度为一维信号序列堆叠为  32 * 32
hidden_layer_sizes = [32, 64, 128]  # LSTM 层数， 每层 神经元个数
attention_dim = hidden_layer_sizes[-1] # embed_dim 默认LSTM输出的维度， 可以修改！
output_dim = 10



