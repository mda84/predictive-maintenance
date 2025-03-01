import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import os

# Load configuration from config.yaml
with open("../data_pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class SensorDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data.fillna(method='ffill', inplace=True)
        # Assume all columns except 'target' are features.
        #self.X = self.data.drop(columns=["target"]).values.astype(np.float32)
        #self.y = self.data["target"].values.astype(np.float32)
        self.X = self.data.drop(columns=["RUL", "scenario"]).values.astype(np.float32)
        self.y = self.data["RUL"].values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Simple LSTM model for time-series prediction.
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_model():
    dataset = SensorDataset("../data/CMAPSSData/sensor_data.csv")
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    input_dim = dataset.X.shape[1]
    model = LSTMModel(input_dim, config["hidden_dim"], 1)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(config["epochs"]):
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            # For simplicity, assume sequence length = 1.
            X_batch = X_batch.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(dataloader):.4f}")
    
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/lstm_model.pth")
    print("Model saved to model/lstm_model.pth")

if __name__ == "__main__":
    train_model()
