import torch
import torch.nn as nn


class MultiModalModel(nn.Module):
    def __init__(self, ip_size_landmarks, ip_size_aus, ip_size_gaze, num_layers, hidden_size, output_size):
        super(MultiModalModel, self).__init__()

        self.lstm_landmarks = nn.LSTM(ip_size_landmarks, hidden_size, num_layers)
        self.lstm_aus = nn.LSTM(ip_size_aus, hidden_size, num_layers)
        self.lstm_gaze = nn.LSTM(ip_size_gaze, hidden_size, num_layers)

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
