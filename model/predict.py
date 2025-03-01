import torch
import numpy as np
from model.train import LSTMModel

def predict(input_features, model_path="model/lstm_model.pth"):
    # Convert input_features (list) to a NumPy array.
    input_features = np.array(input_features, dtype=np.float32)
    input_features = input_features.reshape(1, -1)  # shape: (1, features)
    input_features = torch.tensor(input_features).unsqueeze(1)  # shape: (1, seq=1, features)
    input_dim = input_features.shape[-1]
    model = LSTMModel(input_dim, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        prediction = model(input_features)
    return prediction.item()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py feature1 feature2 ...")
    else:
        features = list(map(float, sys.argv[1:]))
        pred = predict(features)
        print("Predicted failure probability:", pred)
