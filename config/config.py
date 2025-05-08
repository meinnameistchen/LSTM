# -*- coding: utf-8 -*-
import torch
# �豸����
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ģ������
MODEL_SAVE_PATH = "saved_models/best_model.pt"
LOSS_SAVE_PATH = 'results/loss_result.png'
class_names_dict = {
    0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5",
    5: "C6", 6: "C7", 7: "C8", 8: "C9", 9: "C10"
}
# ��ȡΪ list�����ռ�˳��
class_names = [class_names_dict[i] for i in range(len(class_names_dict))]
# ������
EPOCHS = 30
LEARNING_RATE = 0.001
# Configuration file for CNN1DModel
batch_size = 32
input_dim = 32   # ����ά��Ϊһά�ź����жѵ�Ϊ  32 * 32
hidden_layer_sizes = [32, 64, 128]  # LSTM ������ ÿ�� ��Ԫ����
attention_dim = hidden_layer_sizes[-1] # embed_dim Ĭ��LSTM�����ά�ȣ� �����޸ģ�
output_dim = 10



