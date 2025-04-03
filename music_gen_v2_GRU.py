import os
import mido
import numpy as np
from mido import Message, MidiFile, MidiTrack
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. 预处理

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

notes_scaled = [list(note) for note in notes_scaled]

n_prev = 30
X_list, y_list = [], []
for i in range(len(notes_scaled) - n_prev):
    X_list.append(notes_scaled[i : i + n_prev])
    y_list.append(notes_scaled[i + n_prev])

X_test = X_list[-300:]
X_train = X_list[:-300]
y_train = y_list[:-300]
y_test = y_list[-300:] 

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test  = np.array(X_test, dtype=np.float32)
y_test  = np.array(y_test, dtype=np.float32)

# 2. 创建 PyTorch 数据加载器

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. 定义 GRU 模型（repalce LSTM ）

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=1, hidden_size=256, batch_first=True)
        self.dropout1 = nn.Dropout(0.6)
        self.gru2 = nn.GRU(input_size=256, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(0.6)
        self.gru3 = nn.GRU(input_size=128, hidden_size=64, batch_first=True)
        self.dropout3 = nn.Dropout(0.6)
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = self.dropout2(out)
        out, _ = self.gru3(out)
        out = out[:, -1, :]
        out = self.dropout3(out)
        out = self.linear(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型

import matplotlib.pyplot as plt

num_epochs = 10
loss_history = []  # 用于记录每个 epoch 的训练损失

for epoch in range(1, num_epochs + 1):
  model.train()
  epoch_loss = 0.0
  for batch_X, batch_y in train_loader:
    # 将数据移动到设备上
    batch_X = batch_X.to(device)  # (batch_size, seq_length, 1)
    batch_y = batch_y.to(device)  # (batch_size, 1)
    
    optimizer.zero_grad()
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()
    
    epoch_loss += loss.item() * batch_X.size(0)
  
  epoch_loss /= len(train_loader.dataset)
  loss_history.append(epoch_loss)  # 记录当前 epoch 的损失
  print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")
  
  # 保存每个epoch结束后的模型检查点
  checkpoint_dir = "./Checkpoints"
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_model_{epoch:02d}.pt")
  torch.save(model.state_dict(), checkpoint_path)

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()
plt.show()

# 保存训练损失曲线到本地
loss_curve_path = "GRU_tl_curve.png"
plt.savefig(loss_curve_path)
print(f"Training loss curve saved to {loss_curve_path}")

# =====================
# 5. 模型评估、预测与 MIDI 文件生成
# =====================

model.eval()
with torch.no_grad():
    # 将测试集数据转换为 tensor 并移动至设备
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # 计算测试集上的预测以及 MSE 损失
    test_predictions_tensor = model(X_test_tensor)
    test_loss = criterion(test_predictions_tensor, y_test_tensor)
    print(f"Test Loss (MSE): {test_loss.item():.4f}")
    
    # 得到模型预测结果，并移回 CPU
    test_predictions = test_predictions_tensor.cpu().numpy().squeeze()

# 对预测结果进行反归一化处理，并四舍五入转换为整数，作为 MIDI 的 note 值
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
    # 可以再重复一次 note_off 来使节奏更加清晰
    track.append(msg_off)
mid_new.tracks.append(track)
mid_new.save("GRU_music.mid")