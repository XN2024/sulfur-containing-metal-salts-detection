# coding: utf-8

from Transformer import Transformer
from LSTM import LSTM
from GRU import GRU
from RNN import RNN
from CNN import CNN
from Transformer import Transformer
from TextCNN import TextCNN
from TextRCNN import TextRCNN
from FastText import FastText
from DPCNN import DPCNN
# from MLP_NP import MLP
# from model_final import MLP
from collections import deque
import numpy as np
import os
import torch
from math import sqrt
import torch.nn as nn
import csv
import datetime
import pandas as pd
from input_final import InputData
# import gensim
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
from sklearn.metrics import accuracy_score,classification_report, f1_score, precision_score, recall_score, roc_auc_score
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def train(data_address,data_test_address, out_csv_path,data_name,output_dir,latest_checkpoint_file,start_record_epoch,vector_address=None, embd_dimension=2, embd_dimension_suffix=5, hidden_dim_suffix=2,
          train_splitThreshold=0.8,
          time_unit='day', batch_size=9,
          loss_type='L1Loss', optim_type='Adam', model_type='RNN', hidden_dim=10,    ######################修改model_type名，分别是RNN，LSTM和GRU
          train_type='iteration', n_layer=1, dropout=0.1, max_epoch_num=10, learn_rate_min=0.0001, path_length=8,
          train_record_folder='./train_record/', model_save_folder='./model/', result_save_folder='./result/',
          ts_type='set'):
    # 初始化数据
    learn_rate = 0.01
    out_size = 1

    # print("len(train_singlePrefixData1[0]):    $$$$$$$$$", len(train_singlePrefixData1[0]))

    if loss_type == 'L1Loss':
        criterion = nn.L1Loss()
        # criterion = nn.CrossEntropyLoss()
    elif loss_type == 'MSELoss':
        criterion = nn.MSELoss()


    # 开始训练
    data = pd.read_excel(data_address).values
    data1=data[:,0:6]
    label=data[:,6]

    tensor_x = torch.from_numpy(data1.astype(np.float32))
    tensor_y = torch.from_numpy(label.astype(np.float32))
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size, drop_last=True, shuffle=False)

    if model_type == 'GRU':
        model=GRU(len(data1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer ,dropout=dropout)
    elif model_type == 'RNN':
        model = RNN(len(data1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout)
    elif model_type == 'LSTM':
        model = LSTM(len(data1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout)
    elif model_type == 'CNN':
        model = CNN(len(data1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout)
    elif model_type == 'TF':
        model = Transformer(len(data1[0]), nhead=1,dim_feedforward=hidden_dim,
                     num_layers=n_layer, dropout=dropout)
    elif model_type == 'TCNN':
        model = TextCNN(len(data1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout)
    elif model_type == 'TRCNN':
        model = TextRCNN(len(data1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout)
    elif model_type == 'FT':
        model = FastText(len(data1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout)
    elif model_type == 'DPCNN':
        model = DPCNN(len(data1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    data_test = pd.read_excel(data_test_address).values
    data1_test = data_test[:, 0:6]
    label_test = data_test[:, 6]
    tensor_x1 = torch.from_numpy(data1_test.astype(np.float32))
    tensor_y1 = torch.from_numpy(label_test.astype(np.float32))
    testPre_dataset = torch.utils.data.TensorDataset(tensor_x1, tensor_y1)
    testPre_dataset_loader = DataLoader(testPre_dataset, batch_size, drop_last=True, shuffle=False)

    #     训练
    # model = MLP()
    for enpoch in range(100):
        total_loss = torch.FloatTensor([0])
        for i, (input, target) in enumerate(my_dataset_loader):
            optimizer.zero_grad()
            # print("input:              ", np.array(input).shape)
            target=target.unsqueeze(1)
            output = model(Variable(input))
            loss = criterion(output, target)
            # optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.data
        predList=[]
        realList=[]
        pre_all=[]
        real_all=[]
        for j, (input1, target1) in enumerate(testPre_dataset_loader):
            # print("input:              ", np.array(input).shape)
            input1=input1.float()
            target1=target1.float().unsqueeze(1)
            target2=target1.detach().numpy()
            output = model(input1)
            output1 = output.detach().numpy()
            predList += [pdic1.item() for pdic1 in output1]
            realList += [pdic2.item() for pdic2 in target1]

            for item1 in target2:
                real_all.append(item1)
            for line in range(len(output1)):
                if output1[line][0] < 0.5:
                    output1[line][0] = 0
                if output1[line][0] >= 0.5 and output1[line][0] < 1.5:
                    output1[line][0] = 1
                if output1[line][0] >= 1.5 and output1[line][0] < 2.5:
                    output1[line][0] = 2
                if output1[line][0] >= 2.5 and output1[line][0] < 3.5:
                    output1[line][0] = 3
                if output1[line][0] >= 3.5:
                    output1[line][0] = 4
            for item2 in output1:
                pre_all.append(item2)
        pre_final=np.squeeze(np.array(pre_all))
        real_final=np.squeeze(np.array(real_all))

        class_accuracies = []
        for i in range(len(set(real_final))):
            class_indices = np.where(np.array(real_final) == i)[0]
            y_true_i = np.array([real_final[j] for j in class_indices])
            y_pred_i = np.array([pre_final[j] for j in class_indices])

            acc = accuracy_score(y_true_i, y_pred_i)
            class_accuracies.append((i, acc))
        acc = accuracy_score(real_final, pre_final)
        f1_macro = f1_score(real_final, pre_final,
                            average='macro', zero_division=1)
        f1_micro = f1_score(real_final, pre_final,
                            average='micro', zero_division=1)
        prec_micro = precision_score(real_final, pre_final, average='micro', zero_division=1)
        prec_macro = precision_score(real_final, pre_final, average='macro', zero_division=1)

        recall_micro = recall_score(real_final, pre_final, average='micro', zero_division=1)
        recall_macro = recall_score(real_final, pre_final, average='macro', zero_division=1)

        acc_every_class = classification_report(real_final, pre_final)
        with open(out_csv_path, 'a+', newline='') as file:
            f_csv = csv.writer(file)
            row = []
            row.append(acc)
            row.append(f1_macro)
            row.append(f1_micro)
            row.append(prec_micro)
            row.append(prec_macro)
            row.append(recall_micro)
            row.append(recall_macro)
            row.append(class_accuracies)
            row.append(acc_every_class)
            f_csv.writerow(row)
            row.clear()
        if enpoch % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        scheduler.step()
        if enpoch==99:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": enpoch,
                    "lr": scheduler.get_last_lr()

                },
                os.path.join(output_dir, latest_checkpoint_file),
            )
        # print("@@@@@@@@@@@@@@@@@@@@@@:   ", total_loss)

def computeMAE(list_a, list_b):
    MAE_temp = []
    for num in range(len(list_a)):
        MAE_temp.append(abs(list_a[num] - list_b[num]))
    MAE = sum(MAE_temp) / len(list_a)
    return MAE


def computeMSE(list_a, list_b):
    MSE_temp = []
    for num in range(len(list_a)):
        MSE_temp.append((list_a[num] - list_b[num]) * (list_a[num] - list_b[num]))
    MSE = sum(MSE_temp) / len(list_a)
    return MSE


def computeTOTAL(list_a, list_b):
    TOTAL_temp = []
    for num in range(len(list_a)):
        TOTAL_temp.append(abs(list_a[num] - list_b[num]))
    TOTAL = sum(TOTAL_temp)
    return TOTAL


def computeMEAN(list_a, list_b):
    MEAN_temp = []
    for num in range(len(list_a)):
        MEAN_temp.append(abs(list_a[num] - list_b[num]))
    MEAN = sum(MEAN_temp) / len(list_a)
    return MEAN




if __name__ == '__main__':
    # #       BPIC2017A.csv   BPIC2017O1.csv    BPIC2017W.csv
    # BPIC_2019_A11.csv   BPIC_2019_O.csv   BPIC_2019_W1.csv
    train(data_address='./单一离子体系一TF_train.xlsx',  # 修改数据名
          data_test_address='./单一离子体系一TF_test.xlsx',
          out_csv_path= './体系一_result_baseline/TF_danyilizi_result.csv',
          data_name='result5_1%', embd_dimension=1, ################################################## 修改数据名
          train_splitThreshold=0.8, #这是为了把所有数据都训练模型，目的是保存隐状态。
          batch_size=9, n_layer=1, hidden_dim=1,
          loss_type='L1Loss', optim_type='Adam', model_type='TF', ######################修改model_type名，分别是RNN，LSTM和GRU
          train_type='iteration',output_dir='./model',latest_checkpoint_file='TF_test_danyilizi',start_record_epoch=0)
    # test(data_address='./Data/result10_50%.csv',  # 修改数据名
    #       data_name='result10', embd_dimension=1,  ################################################## 修改数据名
    #       train_splitThreshold=0.8,  # 这是为了把所有数据都训练模型，目的是保存隐状态。
    #       batch_size=9, n_layer=1, hidden_dim=1,
    #       loss_type='L1Loss', optim_type='Adam', model_type='GRU',  ######################修改model_type名，分别是RNN，LSTM和GRU
    #       train_type='iteration', output_dir='./model_ablation', latest_checkpoint_file='GRU_3inlayer_result10_50%.pt')