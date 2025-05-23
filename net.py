import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(params.input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, send):
        send = self.fc1(send)
        send = self.relu(send)
        send = self.fc2(send)
        send = self.relu(send)
        send = self.fc3(send)
        return self.sigmoid(send)
        

def loss_fn(outputs, labels):
    return nn.BCELoss()(outputs, labels)


def accuracy(outputs, labels):
    with torch.no_grad():  # 关闭梯度计算
        y_pred_class = (outputs > 0.5).float()  # 将概率值转换为类别标签
        accuracy = (y_pred_class == labels).float().mean().item()  # 计算准确率
        return accuracy


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
