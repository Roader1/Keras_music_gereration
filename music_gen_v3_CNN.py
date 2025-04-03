import os
import mido
import numpy as np
from mido import Message, MidiFile, MidiTrack
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =====================
# 1. 数据预处理
# =====================

# 读取 MIDI 文件（请确保路径正确）
mid = MidiFile("Samples/Nintendo_-_Pokemon_Fire_Red_Route_1_Piano_Cover_Hard_Version.mid")
notes = []
for msg in mid:
    # 只筛选 channel 为 0 且类型为 note_on 的消息
    if not msg.is_meta and msg.channel == 0 and msg.type == "note_on":
        data = msg.bytes()
        notes.append(data[1])

# 对 note 值进行归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))
notes_arr = np.array(notes).reshape(-1, 1)
scaler.fit(notes_arr)
notes_scaled = scaler.transform(notes_arr)

# 将每个 note 转换为列表形式（构成每个时间步只包含 1 个特征的一维向量）
notes_scaled = [list(note) for note in notes_scaled]

# 将序列数据分割为训练数据 (X, y) ，其中 X 为连续 n_prev 个 note 的序列，y 为下一个 note
n_prev = 30
X_list, y_list = [], []
for i in range(len(notes_scaled) - n_prev):
    X_list.append(notes_scaled[i : i + n_prev])
    y_list.append(notes_scaled[i + n_prev])

# 分割数据：将最后 300 组数据作为测试数据，其余作为训练数据
X_test = X_list[-300:]
X_train = X_list[:-300]
y_train = y_list[:-300]
y_test = y_list[-300:]  # 补充测试集的目标值

# 转换为 numpy 数组，并保证数据类型为 float32（以便后续转换为 torch.Tensor）
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test  = np.array(X_test, dtype=np.float32)
y_test  = np.array(y_test, dtype=np.float32)

# =====================
# 2. 创建 PyTorch 数据加载器
# =====================

# 将 numpy 数组转换为 PyTorch Tensor，并封装到 TensorDataset 中
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =====================
# 3. 定义 CNN 模型
# =====================
# 为了处理形状为 (batch, seq_length, 1) 的数据，CNN 模型内部先将数据转置为 (batch, channels, seq_length)
# 此处我们采用两层 Conv1d 与 ReLU 激活函数，再经过 MaxPool1d 降采样，最后 Flatten 后通过全连接层输出一个标量
class CNNModel(nn.Module):
    def __init__(self, seq_length=30):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool  = nn.MaxPool1d(kernel_size=2)  # 将时序长度减半
        self.dropout = nn.Dropout(0.5)
        # 经过池化层后的时序长度 = seq_length // 2
        self.fc = nn.Linear(64 * (seq_length // 2), 1)

    def forward(self, x):
        # x 的形状: (batch, seq_length, 1) -> 转换为 (batch, 1, seq_length)
        x = x.transpose(1, 2)
        x = self.conv1(x)      # (batch, 128, seq_length)
        x = self.relu1(x)
        x = self.conv2(x)      # (batch, 64, seq_length)
        x = self.relu2(x)
        x = self.pool(x)       # (batch, 64, seq_length//2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten 成 (batch, 64 * (seq_length//2))
        x = self.fc(x)         # 输出 (batch, 1)
        return x

# 设置设备（优先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(seq_length=n_prev).to(device)

# 定义均方误差损失函数以及 Adam 优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# 4. 训练模型
# =====================

num_epochs = 10 
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        # 将数据移到设备上
        batch_X = batch_X.to(device)  # (batch, seq_length, 1)
        batch_y = batch_y.to(device)  # (batch, 1)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_X.size(0)
    
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # 保存模型检查点
    checkpoint_dir = "./Checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_model_{epoch:02d}.pt")
    torch.save(model.state_dict(), checkpoint_path)

# =====================
# 5. 模型评估、预测与 MIDI 文件生成
# =====================

model.eval()
with torch.no_grad():
    # 将测试集数据转换为 Tensor 并移入设备
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # 使用模型预测测试集数据，并计算测试损失
    test_predictions_tensor = model(X_test_tensor)
    test_loss = criterion(test_predictions_tensor, y_test_tensor)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")
    
    # 从 tensor 转换为 numpy 数组
    test_predictions = test_predictions_tensor.cpu().numpy().squeeze()

# 对预测结果做反归一化处理，并四舍五入转换为整数，作为 MIDI 的 note 值
predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).squeeze()
predictions_int = [int(round(num)) for num in predictions]

# 生成新的 MIDI 文件
mid_new = MidiFile()
track = MidiTrack()
for note in predictions_int:
    # 构造 note_on 消息
    msg_on = Message.from_dict({
        'type': 'note_on',
        'channel': 0,
        'note': note,
        'velocity': 67,
        'time': 0
    })
    # 构造 note_off 消息，设置一定的时长以制造停顿
    msg_off = Message.from_dict({
        'type': 'note_off',
        'channel': 0,
        'note': note,
        'velocity': 67,
        'time': 64
    })
    track.append(msg_on)
    track.append(msg_off)
    track.append(msg_off)
mid_new.tracks.append(track)
mid_new.save("CNN_music.mid")