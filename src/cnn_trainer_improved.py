# cnn_trainer_improved.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
import os
from src.utils import evaluate_model, print_metrics

class CNNTrainerImproved:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 weight_decay: float = 1e-5, device: Optional[str] = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 100, early_stopping_patience: int = 10,
            save_path: str = 'models/cnn/best_model.pth', verbose: bool = True) -> Dict[str, List[float]]:
        best_val_loss = float('inf')
        patience = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if verbose:
                print(f"Epoka {epoch+1}: Train={train_loss:.5f}, Val={val_loss:.5f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    print("Early stopping")
                    break
        self.model.load_state_dict(torch.load(save_path))
        return {'train_loss': self.train_losses, 'val_loss': self.val_losses}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_pred = self.model(X_tensor).cpu().numpy()
        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray, scaler=None) -> Dict[str, float]:
        y_pred = self.predict(X)
        metrics = evaluate_model(y, y_pred, scaler)
        print_metrics(metrics)
        return metrics
