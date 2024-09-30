import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLPHead(nn.Module):
    def __init__(self, input_size, fingerprint_size, hidden_dim):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(2 * input_size + 2 * fingerprint_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class SNN(nn.Module):
    def __init__(self, input_size, fingerprint_size, hidden_dim):
        super(SNN, self).__init__()
        self.input_size = input_size
        self.fingerprint_size = fingerprint_size
        self.hidden_dim = hidden_dim
        self.mlp_head = MLPHead(input_size, fingerprint_size, hidden_dim)

    def forward(self, pair_features):
        output = self.mlp_head(pair_features)
        return output

def train_model(model, train_loader, criterion, optimizer, num_epochs=20, model_save_path=None):
    model.train()
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        all_outputs = []

        for pair_features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(pair_features)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())
            predicted_labels = (outputs > 0.5).float()
            all_predictions.extend(predicted_labels.detach().cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_outputs)

        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, "
                     f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                     f"F1 Score: {f1:.4f}, AUC: {auc:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Saved best model with accuracy: {best_accuracy:.4f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import logging
import torch

def evaluate_model(model, test_loader, output_file):
    model.eval()
    all_labels = []
    all_predictions = []
    all_outputs = []

    with torch.no_grad():
        for pair_features, labels in test_loader:
            if not isinstance(pair_features, torch.Tensor):
                pair_features = torch.tensor(pair_features)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            outputs = model(pair_features)

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())
            predicted_labels = (outputs > 0.5).float()
            all_predictions.extend(predicted_labels.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_outputs = np.array(all_outputs)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    result = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    with open(output_file, 'w') as f:
        for metric, value in result.items():
            f.write(f"{metric}: {value:.4f}\n")

    logging.info(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                 f"F1 Score: {f1:.4f}")
