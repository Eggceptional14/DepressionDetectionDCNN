import torch
import torch.nn as nn

from Attention import Attention


class CNNModel(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size, dropout):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, hidden_size, kernel_size=3, padding=1)
        
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.dropout = nn.Dropout(p=dropout)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.attention = Attention(hidden_size, 1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(x.shape)

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.dropout(x)

        # x = x.permute(0, 2, 1)
        # x, attention_weights = self.attention(x)
        # x = x.mean(dim=1)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dropout_prob, num_layers=1):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.attention = Attention(hidden_size, 1)

        # self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, output_size),
        )

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
    def __init__(self, ip_size_landmarks, ip_size_aus, ip_size_gaze, hidden_size, output_size, device, num_layers=1):
        super(MultiModalModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm_landmarks = nn.LSTM(ip_size_landmarks, hidden_size, num_layers, batch_first=True)
        self.lstm_aus = nn.LSTM(ip_size_aus, hidden_size, num_layers, batch_first=True)
        self.lstm_gaze = nn.LSTM(ip_size_gaze, hidden_size, num_layers, batch_first=True)

        self.attention_landmarks = Attention(hidden_size, 1)
        self.attention_aus = Attention(hidden_size, 1)
        self.attention_gaze = Attention(hidden_size, 1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, landmarks, aus, gaze):
        h0_landmarks = torch.zeros(self.num_layers, landmarks.size(0), self.hidden_size).to(self.device)
        c0_landmarks = torch.zeros(self.num_layers, landmarks.size(0), self.hidden_size).to(self.device)
        h0_aus = torch.zeros(self.num_layers, aus.size(0), self.hidden_size).to(self.device)
        c0_aus = torch.zeros(self.num_layers, aus.size(0), self.hidden_size).to(self.device)
        h0_gaze = torch.zeros(self.num_layers, gaze.size(0), self.hidden_size).to(self.device)
        c0_gaze = torch.zeros(self.num_layers, gaze.size(0), self.hidden_size).to(self.device)

        # Process each modality through its LSTM
        out_landmarks, _ = self.lstm_landmarks(landmarks, (h0_landmarks, c0_landmarks))
        out_aus, _ = self.lstm_aus(aus, (h0_aus, c0_aus))
        out_gaze, _ = self.lstm_gaze(gaze, (h0_gaze, c0_gaze))

        out_landmarks, attn_weights_landmarks = self.attention_landmarks(out_landmarks)
        out_aus, attn_weights_aus = self.attention_aus(out_aus)
        out_gaze, attn_weights_gaze = self.attention_gaze(out_gaze)

        out_landmarks = out_landmarks[:, -1, :]
        out_aus = out_aus[:, -1, :]
        out_gaze = out_gaze[:, -1, :]

        combined_out = torch.cat((out_landmarks, out_aus, out_gaze), dim=1)

        return self.fc(combined_out), (attn_weights_landmarks, attn_weights_aus, attn_weights_gaze)