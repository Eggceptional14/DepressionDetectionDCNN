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


def train_model(config):
    # use the same config as the previous work
    landmarks_size = 136
    aus_size = 1
    gaze_size = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(input_size=landmarks_size, hidden_size=config['hidden_size'], output_size=2, device=device)
    model.to(device)

    # train_labels_dir = "data/daic/train_split_Depression_AVEC2017.csv"
    # val_labels_dir = "data/daic/dev_split_Depression_AVEC2017.csv"
    train_labels_dir = "/Users/pitchakorn/Dissertation/data/daic/train_split_Depression_AVEC2017.csv"
    val_labels_dir = "/Users/pitchakorn/Dissertation/data/daic/dev_split_Depression_AVEC2017.csv"

    train_dataset = DAICDataset(train_labels_dir, is_train=True)
    val_dataset = DAICDataset(val_labels_dir, is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    if checkpoint := train.get_checkpoint():
        with checkpoint.as_directory() as checkpoint_dir:
          with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
            checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    criterion = nn.CrossEntropyLoss()

    # print("##### Model Training #####\n\n")

    epochs = 10
    for epoch in range(start_epoch, epochs):
        model.train()

        # Training loop
        for batch in train_loader:
            # print("##### Start Training #####")
            pid, landmarks, aus, gaze, label = batch['pid'], batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            output, attention_weights = model(landmarks)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            # print(f'##### {config['batch_size']} Batch Finish #####\n')

        # print('##### Training Complete, Starting Evaluation #####\n')

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_elements = 0

        with torch.no_grad():
            for batch in val_loader:
                # print("##### Start Evaluation #####")
                pid, landmarks, aus, gaze, label = batch['pid'], batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device), batch['label'].to(device)
                output, attention_weights = model(landmarks)
                val_loss += criterion(output, label).item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == label).sum().item()
                total_elements += label.nelement()
                # print(f'##### {config['batch_size']} Batch Finish #####\n')

        val_loss /= len(val_loader)
        accuracy = total_correct / total_elements

        # print(f'Validation Loss: {val_loss} | Accuracy: {accuracy}')

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
          with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:
            pickle.dump(checkpoint_data, fp)

          checkpoint = Checkpoint.from_directory(checkpoint_dir)

          train.report(
              {"loss": val_loss, "accuracy": accuracy},
              checkpoint=checkpoint,
          )

    # print('##### Training complete #####\n\n')

if __name__ == "__main__":
    config = {
        "dropout": 0.2,
        "lr": 0.001,
        "batch_size": 1,
        "hidden_size": 32
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
            train_model,
            resources_per_trial={"cpu": 4, "gpu": 0},
            config=config,
            num_samples=1,
            scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # save best model
    best_checkpoint = best_trial.checkpoint

    with best_checkpoint.as_directory() as checkpoint_dir:
        best_model_state = torch.load(os.path.join(checkpoint_dir, 'data.pkl'))

    model = LSTMModel(input_size=136, hidden_size=best_trial.config['hidden_size'], output_size=2, device="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(best_model_state["net_state_dict"])

    torch.save(model.state_dict(), "best_model.pth")