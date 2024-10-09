import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Đọc file CSV
file_path = "data_body_12classes.csv"  # Đường dẫn đến file CSV
df = pd.read_csv(file_path)

# Tách nhãn (cột đầu tiên) và dữ liệu (tọa độ keypoint x, y)
X = df.iloc[:, 1:].values  # Dữ liệu tọa độ keypoint
y = df.iloc[:, 0].values   # Nhãn hành động

# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# print(X)
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
#
#
# print("\n")
# print("\n")
#
# print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
# print(y)
# print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")


# # Chuẩn hóa dữ liệu tọa độ
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# print("\n")
# print("\n")
# print("SCALEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
# print(X_scaled)
# print("SCALEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")


# Định nghĩa Dataset và DataLoader cho PyTorch
class KeypointDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Định nghĩa độ dài chuỗi (sequence_length)
sequence_length = 40  # Số frame liên tiếp trong một chuỗi


# Hàm tạo sequence từ dữ liệu
def create_sequences(data, labels, sequence_length):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - sequence_length):
        if labels[i] == labels[i + sequence_length - 1]:
            # Tạo một sequence với độ dài sequence_length
            sequence = data[i:i+sequence_length]
            sequences.append(sequence)
            # Nhãn của sequence là nhãn của frame cuối cùng trong chuỗi
            sequence_labels.append(labels[i + sequence_length - 1])
        else:
            continue
    return np.array(sequences), np.array(sequence_labels)


# Tạo sequences từ dữ liệu huấn luyện và kiểm tra
X, y = create_sequences(X, y, sequence_length)

# print("\n")
# print("\n")
# print("XTRAINSEQUENCEEEEEEEEEEEEEEEEEE")
# print(X)
# print("XTRAINSEQUENCEEEEEEEEEEEEEEEEEE")
# print("\n")
# print("\n")
# print("YTRAINSEQUENCEEEEEEEEEEEEEEEEEE")
# print(y)
# print("YTRAINSEQUENCEEEEEEEEEEEEEEEEEE")


# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X, y, test_size=0.2, random_state=42)


# Chuyển đổi dữ liệu thành tensor của PyTorch
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.long)

# Định nghĩa Dataset và DataLoader cho PyTorch
train_dataset = KeypointDataset(X_train_tensor, y_train_tensor)
test_dataset = KeypointDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Định nghĩa lớp LSTM kế thừa từ nn.Module
class LSTMActionRecognition(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMActionRecognition, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Định nghĩa lớp LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        # Decode the hidden state of the last time step
        out = self.fc(out)  # Lấy output của time step cuối cùng
        return out


# Định nghĩa hyperparameters
input_size = 34  # Số cột (tọa độ x, y cho 17 keypoints)
hidden_size = 64  # Số chiều hidden của LSTM
output_size = 12  # Số class (12 hành động)
num_layers = 2  # Số lớp LSTM
num_epochs = 40  # Số epoch
learning_rate = 0.001  # Learning rate

# Khởi tạo mô hình, loss function và optimizer
model = LSTMActionRecognition(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Huấn luyện mô hình
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        # inputs có kích thước (batch_size, sequence_length, input_size)
        labels = labels

        # Forward pass
        outputs = model(inputs)  # Đưa sequence vào mô hình
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Kiểm tra trên tập test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Lưu lại state_dict của mô hình sau khi huấn luyện xong
torch.save(model.state_dict(), 'lstm_action_recognition_40_epochs.pth')
print("Model saved successfully!")
