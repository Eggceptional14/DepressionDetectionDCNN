import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        output, attention_weights = self.attention(x, x, x)
        return output, attention_weights

class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob, device):
        super(LSTMModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size, 4)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out, attention_weights = self.attention(out)

        out = torch.mean(out, dim=1)

        out = self.dropout(out)
        
        return out, attention_weights

class CNNModule(nn.Module):
    def __init__(self, input_channels, dropout):
        super(CNNModule, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=dropout)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.attention = Attention(256, 4)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.relu1(self.conv1(x))
        # print('Shape before pool', 1, x.shape)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.relu2(self.conv2(x))
        # print('Shape before pool', 2, x.shape)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.relu3(self.conv3(x))
        # print('Shape before pool', 3, x.shape)
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x, attention_weights = self.attention(x)

        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        return x, attention_weights

class CNNModel(nn.Module):
    def __init__(self, input_channels, output_size, dropout):
        super(CNNModel, self).__init__()

        self.cnn_module = CNNModule(input_channels, dropout)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        x, attention_weights = self.cnn_module(x)
        x = self.fc(x)
        return x, attention_weights

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dropout_prob, num_layers=1):
        super(LSTMModel, self).__init__()

        self.lstm_module = LSTMModule(input_size, hidden_size, num_layers, dropout_prob, device)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        x, attention_weights = self.lstm_module(x)
        x = self.fc(x)
        return x, attention_weights

class MultiModalModel(nn.Module):
    def __init__(self, ip_size_landmarks, ip_size_aus, ip_size_gaze, hidden_size, output_size, device, dropout_prob, num_layers=1):
        super(MultiModalModel, self).__init__()
        
        self.landmarks_lstm = LSTMModule(ip_size_landmarks, hidden_size, num_layers, dropout_prob, device)
        
        self.aus_cnn = CNNModule(ip_size_aus, dropout_prob)
        self.gaze_cnn = CNNModule(ip_size_gaze, dropout_prob)

        self.combined_attention = Attention(hidden_size + 256 * 2, num_heads=4)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, output_size),
        )

    def forward(self, landmarks, aus, gaze):
        out_landmarks, attn_weights_landmarks = self.landmarks_lstm(landmarks)
        
        out_aus, attn_weights_aus = self.aus_cnn(aus)
        out_gaze, attn_weights_gaze = self.gaze_cnn(gaze)

        combined_out = torch.cat((out_landmarks, out_aus, out_gaze), dim=1).unsqueeze(1)

        # combined_out, combined_attention_weights = self.combined_attention(combined_out)

        combined_out = combined_out.squeeze(1)

        out = self.fc(combined_out)
        
        return out, (attn_weights_landmarks, attn_weights_aus, attn_weights_gaze)
