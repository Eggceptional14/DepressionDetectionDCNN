import torch
import torch.nn as nn

from Attention import Attention


class CNNModel(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size, num_heads):
        super(CNNModel, self).__init__()

        # TODO: CNN architecture
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, hidden_size, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dropout_prob, num_layers=1):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.attention = SelfAttention(hidden_size)
        self.attention = Attention(hidden_size, 1)

        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_prob)

        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, output_size),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out, attention_weights = self.attention(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return self.softmax(out), attention_weights
    
class MultiModalModel(nn.Module):
    def __init__(self, ip_size_landmarks, ip_size_aus, ip_size_gaze, hidden_size, output_size, num_layers=1):
        super(MultiModalModel, self).__init__()

        self.lstm_landmarks = nn.LSTM(ip_size_landmarks, hidden_size, num_layers, batch_first=True)
        self.lstm_aus = nn.LSTM(ip_size_aus, hidden_size, num_layers, batch_first=True)
        self.lstm_gaze = nn.LSTM(ip_size_gaze, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

    def forward(self, landmarks, aus, gaze):
        _, (landmark_out, _) = self.lstm_landmarks(landmarks)
        _, (aus_out, _) = self.lstm_aus(aus)
        _, (gaze_out, _) = self.lstm_gaze(gaze)

        # Concatenate LSTM outputs
        combined = torch.cat((aus_out[-1], landmark_out[-1], gaze_out[-1]), dim=1)

        return self.fc(combined)