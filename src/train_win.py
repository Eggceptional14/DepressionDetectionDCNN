import os, gc
import tempfile
import pandas as pd
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from Model import CNNModel, LSTMModel, MultiModalModel
from Dataset import DAICDataset

load_dotenv()

def train_model(config):
    landmarks_size = 136
    aus_size = 1
    gaze_size = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = LSTMModel(input_size=landmarks_size, hidden_size=config['hidden_size'], dropout_prob=config['dropout'], output_size=2, device=device)
    model = CNNModel(landmarks_size, config['hidden_size'])
    model.to(device)

    train_labels_dir = r"D:\DepressionDetectionDL\data\train_split_Depression_AVEC2017.csv"
    val_labels_dir = r"D:\DepressionDetectionDL\data\dev_split_Depression_AVEC2017.csv"

    train_dataset = DAICDataset(train_labels_dir, is_train=True)
    val_dataset = DAICDataset(val_labels_dir, is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        torch.cuda.empty_cache()
        gc.collect()
        print(f'Epoch: {epoch}')

        # Training loop
        for i, batch in enumerate(train_loader):
            _, landmarks, label = batch['pid'], batch['landmarks'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            
            with autocast(device_type=device):
                output, _ = model(landmarks)
                loss = criterion(output, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_elements = 0

        with torch.no_grad():
            for batch in val_loader:
                _, landmarks, label = batch['pid'], batch['landmarks'].to(device), batch['label'].to(device)
                output, _ = model(landmarks)
                val_loss += criterion(output, label).item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == label).sum().item()
                total_elements += label.nelement()

        val_loss /= len(val_loader)
        accuracy = total_correct / total_elements

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}, Accuracy: {accuracy}')

    # Save the final model
    torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    config = {
        "dropout": 0.2,
        "lr": 0.001,
        "batch_size": 1,
        "hidden_size": 32
    }

    train_model(config)
