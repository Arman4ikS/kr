import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(14, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.dense1 = nn.Linear(32, 96)
        self.dense2 = nn.Linear(96, 128)
        self.dense3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Последний шаг последовательности
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.dense3(x)
        return x

# Инициализация модели
model = LSTMModel()

# 3. Загрузка весов
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Переводим в режим оценки

scaler = StandardScaler()
data_train = pd.read_csv("train_FD001.txt",sep=" ",header=None)
data_train.drop(columns=[0,26,27],inplace=True)
column_names = ['cycles','setting_1','setting_2','setting_3','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
data_train.columns = column_names
train_data = scaler.fit_transform(data_train.drop(columns = ['setting_1','setting_2','setting_3', 'T2', 'P2','P15','P30', 'epr',
                 'farB', 'Nf_dmd', 'PCNfR_dmd']))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

def process_input_data_with_targets(input_data, target_data=None, window_length=1, shift=1):
    num_batches = int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data = np.repeat(np.nan, repeats=num_batches * window_length * num_features).reshape(num_batches,
                                                                                                window_length,
                                                                                                num_features)
    if target_data is None:
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
        return output_data
    else:
        output_targets = np.repeat(np.nan, repeats=num_batches)
        for batch in range(num_batches):
            output_data[batch, :, :] = input_data[(0 + shift * batch):(0 + shift * batch + window_length), :]
            output_targets[batch] = target_data[(shift * batch + (window_length - 1))]
        return output_data, output_targets

def process_test_data(test_data_for_an_engine, window_length, shift, num_test_windows=1):
    max_num_test_batches = int(np.floor((len(test_data_for_an_engine) - window_length) / shift)) + 1
    if max_num_test_batches < num_test_windows:
        required_len = (max_num_test_batches - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, max_num_test_batches
    else:
        required_len = (num_test_windows - 1) * shift + window_length
        batched_test_data_for_an_engine = process_input_data_with_targets(test_data_for_an_engine[-required_len:, :],
                                                                          target_data=None,
                                                                          window_length=window_length, shift=shift)
        return batched_test_data_for_an_engine, num_test_windows


def processing(unit_id: str):
    engine = create_engine('sqlite:///database.db')
    test_data = pd.read_sql_table(f'unit_ID_{unit_id}', engine)

    test_data.columns = column_names

    window_length = 30
    shift = 1
    num_test_windows = 3
    num_test_machines = 1
    processed_test_data = []
    num_test_windows_list = []
    if len(test_data) > window_length:
        columns_to_be_dropped = ['setting_1', 'setting_2', 'setting_3', 'T2', 'P2', 'P15', 'P30', 'epr',
                                 'farB', 'Nf_dmd', 'PCNfR_dmd']

        test_data = test_data.drop(columns=columns_to_be_dropped)

        test_data = scaler.transform(test_data)

        test_data_first_column = pd.DataFrame(np.array([int(unit_id)]*len(test_data)), columns=['unit_ID'])
        test_data = pd.DataFrame(data=np.c_[test_data_first_column, test_data])

        for i in np.arange(1, num_test_machines + 1):
            temp_test_data = test_data[test_data[0] == unit_id].drop(columns = [0]).values

            test_data_for_an_engine, num_windows = process_test_data(temp_test_data, window_length=window_length,
                                                                     shift=shift,
                                                                     num_test_windows=num_test_windows)

            processed_test_data.append(test_data_for_an_engine)
            num_test_windows_list.append(num_windows)

        test_tensor = torch.FloatTensor(processed_test_data[0])
        model.eval()
        with torch.no_grad():
            rul_pred = model(test_tensor).squeeze().numpy()

        preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])
        mean_pred_for_each_engine = [
            np.average(ruls, weights=np.repeat(1 / num_windows, num_windows))
            for ruls, num_windows in zip(preds_for_each_engine, num_test_windows_list)
        ]
        rul = int(np.mean(mean_pred_for_each_engine[0]))
        return rul
    return 0

