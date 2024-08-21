import torch
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score

from Model import *
from Dataset import DAICDataset


def test(config):    
    feature_size = {"landmarks": 136, "aus": 20, "gaze": 12}
    m_name = f"{config['model']}_{config['feature']}"
    model_path = f'/Users/pitchakorn/Dissertation/models/{m_name}_chunk/best_model/model.pth'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config['model'] == "lstm":
        model = LSTMModel(input_size=feature_size[config['feature']], 
                          hidden_size=config['hidden_size'], 
                          dropout_prob=config['dropout'], 
                          output_size=1, 
                          device=device)
    elif config['model'] == "cnn":
        model = CNNModel(input_channels=feature_size[config['feature']], 
                         hidden_size=config['hidden_size'], 
                         output_size=1, 
                         dropout=config['dropout'])
    elif config['model'] == "multimodal":
        model = CombinationModel(ip_size_landmarks=feature_size['landmarks'],
                                ip_size_aus=feature_size['aus'],
                                ip_size_gaze=feature_size['gaze'],
                                hidden_size=config['hidden_size'],
                                output_size=1,
                                device=device)
        
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    test_labels_dir = "/Users/pitchakorn/Dissertation/data/daic/full_test_split.csv"

    test_dataset = DAICDataset(test_labels_dir, config['chunk_size'], is_train=False)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    criterion = nn.BCELoss()

    model.eval()
    test_loss = 0.0
    all_preds, all_actuals = [], []

    with torch.no_grad():
        for batch in test_loader:
            if config['model'] == "multimodal":
                landmarks, aus, gaze = batch['landmarks'].to(device), batch['aus'].to(device), batch['gaze'].to(device)
                actual = batch['actual'].to(device).float()
                output, _ = model(landmarks, aus, gaze)
            else:
                data = batch[config['feature']].to(device)
                actual = batch['actual'].to(device).float()
                output, _ = model(data)
            
            output = output.squeeze(-1)
            # print(output)
            print(batch['pid'], output, actual)

            loss = criterion(output, actual)
            test_loss += loss.item()

            pred = output.round()
            
            all_preds.extend(pred.cpu().numpy())
            all_actuals.extend(actual.cpu().numpy())

    precision = precision_score(all_actuals, all_preds)
    recall = recall_score(all_actuals, all_preds)
    f1 = f1_score(all_actuals, all_preds)

    avg_loss = test_loss / len(test_loader)
    
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')