import pandas as pd
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ray
import ray.cloudpickle as pickle
from ray import train,tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from Model import CNNModel, LSTMModel, MultiModalModel
from Dataset import DAICDataset

load_dotenv()


def train():
    # use the same config as the previous work
    config = {
        "dropout": 0.2,
        "lr": 0.001,
        "batch_size": 1,
        "hidden_size": 32
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(input_size=136, hidden_size=config['hidden_size'], output_size=1, device=device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    train_labels_dir = "data/daic/train_split_Depression_AVEC2017.csv"
    val_labels_dir = "data/daic/dev_split_Depression_AVEC2017.csv"

    train_dataset = DAICDataset(train_labels_dir, is_train=True)
    val_dataset = DAICDataset(val_labels_dir, is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    print("##### Model Training #####\n\n")

    epochs = 1
    for _ in range(epochs):
        model.train()
        
        # Training loop
        for batch in train_loader:
            print("##### Start Training #####")
            pid, landmarks, aus, gaze, label = batch['pid'], batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            output = model(landmarks)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print(f'##### {config['batch_size']} Batch Finish #####\n')
            
        print('##### Training Complete, Starting Evaluation #####\n')

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_elements = 0

        with torch.no_grad():
            for batch in val_loader:
                print("##### Start Evaluation #####")
                pid, landmarks, aus, gaze, label = batch['pid'], batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device), batch['label'].to(device)
                output = model(landmarks)
                val_loss += criterion(output, label).item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == label).sum().item()
                total_elements += label.nelement()
                print(f'##### {config['batch_size']} Batch Finish #####\n')

        val_loss /= len(val_loader)
        accuracy = total_correct / total_elements

        print(f'Validation Loss: {val_loss} | Accuracy: {accuracy}')

    print('##### Training complete #####\n\n')

        
if __name__ == "__main__":
    train()