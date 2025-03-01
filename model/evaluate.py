import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

class SensorDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data.fillna(method='ffill', inplace=True)
        # Drop non-numeric columns and use "RUL" as target.
        self.X = self.data.drop(columns=["RUL", "scenario"]).values.astype(np.float32)
        self.y = self.data["RUL"].values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def evaluate_model(model_path, csv_file):
    from train import LSTMModel  # Import your model architecture from train.py
    dataset = SensorDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    input_dim = dataset.X.shape[1]
    model = LSTMModel(input_dim, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    mse_loss = 0.0
    count = 0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # Reshape X_batch as needed (for sequence length, etc.)
            X_batch = X_batch.unsqueeze(1)  # assuming sequence length 1 for simplicity
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            mse_loss += loss.item() * X_batch.size(0)
            count += X_batch.size(0)
    mse = mse_loss / count
    print(f"Mean Squared Error on test data: {mse:.4f}")
    return mse

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py model_path csv_file")
    else:
        model_path = sys.argv[1]
        csv_file = sys.argv[2]
        evaluate_model(model_path, csv_file)
