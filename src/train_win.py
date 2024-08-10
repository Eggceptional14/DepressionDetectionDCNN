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

    train_labels_dir = r"D:\DepressionDetectionDL\data\train_split_Depression_AVEC2017.csv"
    val_labels_dir = r"D:\DepressionDetectionDL\data\dev_split_Depression_AVEC2017.csv"

    train_dataset = DAICDataset(train_labels_dir, config['frame_step'], is_train=True)
    val_dataset = DAICDataset(val_labels_dir, config['frame_step'], is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    accs, losses = [], []

    for epoch in range(config['epochs']):
        model.train()
        torch.cuda.empty_cache()
        gc.collect()
        print(f'Epoch: {epoch}')

        for batch in train_loader:
            optimizer.zero_grad()

            if config['model'] == "multimodal":
                landmarks, aus, gaze = batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device)
                label = batch['label'].to(device)
                min_frames = min(len(landmarks), len(aus), len(gaze))
                for i in range(0, min_frames, config['seg_length']):
                        end_landmarks = min(i + config['seg_length'], len(landmarks))
                        end_aus = min(i + config['seg_length'], len(aus))
                        end_gaze = min(i + config['seg_length'], len(gaze))
                        output, attention_weights = model(landmarks[i:end_landmarks],
                                                          aus[i:end_aus],
                                                          gaze[i:end_gaze])
                        loss = criterion(output, label)
                        loss.backward()
                        optimizer.step()
            else:
                data = batch[config['feature']].to(device)
                label = batch['label'].to(device)
                num_frames = len(data)
                for i in range(0, num_frames, config['seg_length']):
                    end = min(i + config['seg_length'], num_frames)
                    output, attention_weights = model(data[i:end])
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
                    min_frames = min(len(landmarks), len(aus), len(gaze))
                    for i in range(0, min_frames, config['seg_length']):
                        end_landmarks = min(i + config['seg_length'], len(landmarks))
                        end_aus = min(i + config['seg_length'], len(aus))
                        end_gaze = min(i + config['seg_length'], len(gaze))
                        output, attention_weights = model(landmarks[i:end_landmarks],
                                                          aus[i:end_aus],
                                                          gaze[i:end_gaze])
                        val_loss += criterion(output, label).item()
                        _, predicted = torch.max(output.data, 1)
                        total_correct += (predicted == label).sum().item()
                        total_elements += label.size(0)
                        
                else:
                    data = batch[config['feature']].to(device)
                    label = batch['label'].to(device)
                    num_frames = len(data)
                    for i in range(0, num_frames, config['seg_length']):
                        end = min(i + config['seg_length'], num_frames)
                        output, attention_weights = model(data[i:end])
                        val_loss += criterion(output, label).item()
                        _, predicted = torch.max(output.data, 1)
                        total_correct += (predicted == label).sum().item()
                        total_elements += label.size(0)

        val_loss /= len(val_loader)
        accuracy = total_correct / total_elements

        losses.append(val_loss)
        accs.append(accuracy)

        print(f'Epoch {epoch}, Validation Loss: {val_loss}, Accuracy: {accuracy}')

        if epoch != 0 and ((epoch % 10 == 0) or (epoch == config['epochs'] - 1)):
            save_dir = os.path.join(r"D:\DepressionDetectionDL\models", m_name, str(epoch))
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, "optim.pth"))
            
    pd.DataFrame({'loss': losses, 'accuracy': accs}).to_csv(f"D:\DepressionDetectionDL\models\{m_name}\progres.csv")
    # Save the final model
    # torch.save(model.state_dict(), f"D:\DepressionDetectionDL\models\{m_name}_{config['epochs']}.pth")

if __name__ == "__main__":
    config = {
        "dropout": 0.2,
        "lr": 0.001,
        "batch_size": 1,
        "hidden_size": 32
    }

    train_model(config)
