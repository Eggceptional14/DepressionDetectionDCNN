import os, gc
import tempfile
import pandas as pd
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Model import CNNModel, LSTMModel, MultiModalModel
from Dataset import DAICDataset

load_dotenv()

def train_model(config):
    feature_size = {"landmarks": 136, "aus": 20, "gaze": 12}
    method = "chunk"
    m_name = f"{config['model']}_{config['feature']}_{method}"

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

    train_dataset = DAICDataset(train_labels_dir, config['chunk_size'], is_train=True)
    val_dataset = DAICDataset(val_labels_dir, config['chunk_size'], is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    accs, losses = [], []
    best_accuracy = 0.0

    for epoch in range(config['epochs']):
        model.train()
        torch.cuda.empty_cache()
        gc.collect()
        print(f'Epoch: {epoch}')

        epoch_loss = 0.0
        total_correct_train = 0
        total_train_elements = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if config['model'] == "multimodal":
                landmarks, aus, gaze = batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device)
                label = batch['label'].to(device)
                output, attention_weights = model(landmarks, aus, gaze)
            else:
                data = batch[config['feature']].to(device)
                label = batch['label'].to(device)
                output, attention_weights = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Accumulate the loss and accuracy
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_correct_train += (predicted == label).sum().item()
            total_train_elements += label.size(0)

        # Calculate average training loss and accuracy
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = total_correct_train / total_train_elements

        print(f'Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

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
                else:
                    data = batch[config['feature']].to(device)
                    label = batch['label'].to(device)
                    output, attention_weights = model(data)

                val_loss += criterion(output, label).item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == label).sum().item()
                total_elements += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = total_correct / total_elements

        losses.append(avg_val_loss)
        accs.append(accuracy)

        print(f'Epoch {epoch}, Validation Loss: {avg_val_loss}, Accuracy: {accuracy}')
        print(scheduler.get_last_lr())
        
        scheduler.step(avg_val_loss)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_dir = os.path.join(r"D:\DepressionDetectionDL\models", m_name, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_model_dir, "model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(best_model_dir, "optim.pth"))

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
