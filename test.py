import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ultralytics import YOLO
import cv2
import copy
import itertools
import torch.nn.functional as fu


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


model = YOLO('yolov8n-pose.pt')
model.to(device)

# Đọc video từ webcam
cap = cv2.VideoCapture(0)


class LSTMActionRecognition(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, output_size=12, num_layers=2):
        super(LSTMActionRecognition, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Định nghĩa lớp LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Lấy output của time step cuối cùng
        return out


model_lstm = LSTMActionRecognition()
model_lstm.to(device)
model_lstm.load_state_dict(torch.load("lstm_action_recognition_40_epochs.pth"))

model_lstm.eval()
seq_list = []
seq_len = 40

while True:
    print("="*100)
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    results = model(frame, device=device)
    detections = results[0]
    if detections:
        for i, pose in enumerate(detections.keypoints):
            for keypoint in pose:
                land_mark_list = keypoint.xy.cpu().tolist()[0]
                process_lm_list = pre_process_landmark(land_mark_list)
                seq_list.append(process_lm_list)
                if len(seq_list) >= 31:
                    seq_list.pop(0)
                    X_val = torch.tensor([seq_list], dtype=torch.float32)
                    X_val.to(device)
                    with torch.no_grad():  # Không cần tính gradient trong quá trình inference
                        outputs = model_lstm(X_val)
                    probabilities = fu.softmax(outputs, dim=1)
                    index = np.argmax(probabilities.cpu().numpy(), axis=1)

                    print("kq:", probabilities)
                    print("Dự đoán:", index)









