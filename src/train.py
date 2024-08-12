import os, tempfile
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


def train_model(config):  # sourcery skip: low-code-quality
    # use the same config as the previous work
    feature_size = {"landmarks": 136, "aus": 20, "gaze": 12}
    m_name = f"{config['model']}_{config['feature']}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config['model'] == "lstm":
        model = LSTMModel(input_size=feature_size[config['feature']], 
                          hidden_size=config['hidden_size'], 
                          dropout_prob=config['dropout'], 
                          output_size=2, 
                          device=device)
    elif config['model'] == "cnn":
        model = CNNModel(input_channels=feature_size[config['feature']], 
                         hidden_size=config['hidden_size'], 
                         output_size=2, 
                         dropout=config['dropout'])
    elif config['model'] == "multimodal":
        model = MultiModalModel(ip_size_landmarks=feature_size['landmarks'],
                                ip_size_aus=feature_size['aus'],
                                ip_size_gaze=feature_size['gaze'],
                                hidden_size=config['hidden_size'],
                                output_size=2,
                                device=device)

    model.to(device)

    train_labels_dir = "/Users/pitchakorn/Dissertation/data/daic/train_split_Depression_AVEC2017.csv"
    val_labels_dir = "/Users/pitchakorn/Dissertation/data/daic/dev_split_Depression_AVEC2017.csv"

    train_dataset = DAICDataset(train_labels_dir, config['chunk_size'], is_train=True)
    val_dataset = DAICDataset(val_labels_dir, config['chunk_size'], is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    if checkpoint := train.get_checkpoint():
        with checkpoint.as_directory() as checkpoint_dir:
          with open(os.path.join(checkpoint_dir, f'data_{m_name}.pkl'), 'rb') as fp:
            checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, config["epochs"]):
        model.train()

        # Training loop
        for batch in train_loader:
            optimizer.zero_grad()

            if config['model'] == "multimodal":
                landmarks, aus, gaze = batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device)
                label = batch['label'].to(device)
                print(landmarks.shape, aus.shape, gaze.shape)
                output, attention_weights = model(landmarks, aus, gaze)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            else:
                data = batch[config['feature']].to(device)
                label = batch['label'].to(device)
                num_frames = len(data)
                output, attention_weights = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_elements = 0

        with torch.no_grad():
            for batch in val_loader:
                if config['model'] == "multimodal":
                    landmarks, aus, gaze = batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device)
                    label = batch['label'].to(device)
                    output, attention_weights = model(landmarks, aus, gaze)
                    val_loss += criterion(output, label).item()
                    _, predicted = torch.max(output.data, 1)
                    total_correct += (predicted == label).sum().item()
                    total_elements += label.size(0)
                        
                else:
                    data = batch[config['feature']].to(device)
                    label = batch['label'].to(device)
                    output, attention_weights = model(data)
                    val_loss += criterion(output, label).item()
                    _, predicted = torch.max(output.data, 1)
                    total_correct += (predicted == label).sum().item()
                    total_elements += label.size(0)

        val_loss /= len(val_loader)
        accuracy = total_correct / total_elements

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, f'data_{m_name}.pkl'), 'wb') as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)

            train.report(
                {"loss": val_loss, "accuracy": accuracy},
                checkpoint=checkpoint,
            )

    # print('##### Training complete #####\n\n')

