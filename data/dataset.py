# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.io import loadmat
from joblib import dump, load
import sklearn
import torch
import os


def load_and_preprocess_mat_data(mat_dir='matfiles', csv_path='data_12k_10c.csv'):
    file_names = ['0_0.mat','7_1.mat','7_2.mat','7_3.mat','14_1.mat','14_2.mat','14_3.mat','21_1.mat','21_2.mat','21_3.mat']
    data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                    'X185_DE_time','X197_DE_time','X209_DE_time','X222_DE_time','X234_DE_time']
    columns_name = ['de_normal','de_7_inner','de_7_ball','de_7_outer','de_14_inner','de_14_ball','de_14_outer','de_21_inner','de_21_ball','de_21_outer']
    data_12k_10c = pd.DataFrame()
    for index in range(10):
        data = loadmat(os.path.join(mat_dir, file_names[index]))
        dataList = data[data_columns[index]].reshape(-1)
        data_12k_10c[columns_name[index]] = dataList[:119808]
    data_12k_10c.to_csv(csv_path, index=False)
    return csv_path


def split_data_with_overlap(data, time_steps, label, overlap_ratio=0.5):
    stride = int(time_steps * (1 - overlap_ratio))
    samples = (len(data) - time_steps) // stride + 1
    data_list = []
    for i in range(samples):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        temp_data = data[start_idx:end_idx].tolist()
        temp_data.append(label)
        data_list.append(temp_data)
    return pd.DataFrame(data_list, columns=[x for x in range(time_steps + 1)])


def make_datasets(data_file_csv, split_rate=[0.6, 0.2, 0.2]):
    origin_data = pd.read_csv(data_file_csv)
    time_steps = 1024
    overlap_ratio = 0.5
    samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    label = 0
    for column_name, column_data in origin_data.items():
        split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
        label += 1
        samples_data = pd.concat([samples_data, split_data])
    samples_data = sklearn.utils.shuffle(samples_data)
    sample_len = len(samples_data)
    train_len = int(sample_len * split_rate[0])
    val_len = int(sample_len * split_rate[1])
    train_set = samples_data.iloc[0:train_len, :]
    val_set = samples_data.iloc[train_len:train_len + val_len, :]
    test_set = samples_data.iloc[train_len + val_len:, :]
    return train_set, val_set, test_set, samples_data


def make_data_labels(dataframe):
    x_data = dataframe.iloc[:, 0:-1]
    y_label = dataframe.iloc[:, -1]
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64'))
    return x_data, y_label


def prepare_and_save_all_datasets():
    csv_path = load_and_preprocess_mat_data()
    train_set, val_set, test_set, _ = make_datasets(csv_path)
    dump(train_set, 'train_set')
    dump(val_set, 'val_set')
    dump(test_set, 'test_set')

    train_xdata, train_ylabel = make_data_labels(train_set)
    val_xdata, val_ylabel = make_data_labels(val_set)
    test_xdata, test_ylabel = make_data_labels(test_set)

    dump(train_xdata, 'trainX_1024_10c')
    dump(val_xdata, 'valX_1024_10c')
    dump(test_xdata, 'testX_1024_10c')
    dump(train_ylabel, 'trainY_1024_10c')
    dump(val_ylabel, 'valY_1024_10c')
    dump(test_ylabel, 'testY_1024_10c')

    print('Êý¾Ý ÐÎ×´£º')
    print(train_xdata.size(), train_ylabel.size())
    print(val_xdata.size(), val_ylabel.size())
    print(test_xdata.size(), test_ylabel.size())


if __name__ == "__main__":
    prepare_and_save_all_datasets()
