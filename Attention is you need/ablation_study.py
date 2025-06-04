import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

class DIN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, seq_length):
        super(DIN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

        self.user_embedding = nn.Embedding(1000, embed_dim)
        self.item_embedding = nn.Embedding(1000, embed_dim)

        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user_hist, target_item):
        user_hist_embed = self.user_embedding(user_hist)
        target_item_embed = self.item_embedding(target_item).unsqueeze(1)

        att_input = torch.cat([user_hist_embed, target_item_embed.expand(-1, self.seq_length, -1),
                               user_hist_embed - target_item_embed, user_hist_embed * target_item_embed], dim=-1)
        att_weights = self.attention(att_input).squeeze(-1)
        att_weights = torch.softmax(att_weights, dim=-1).unsqueeze(-1)

        user_interest = (user_hist_embed * att_weights).sum(dim=1)

        mlp_input = torch.cat([user_interest, target_item_embed.squeeze(1)], dim=-1)
        output = self.mlp(mlp_input)
        return output

class DINWithoutAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, seq_length):
        super(DINWithoutAttention, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

        self.user_embedding = nn.Embedding(1000, embed_dim)
        self.item_embedding = nn.Embedding(1000, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user_hist, target_item):
        user_hist_embed = self.user_embedding(user_hist)
        target_item_embed = self.item_embedding(target_item).unsqueeze(1)

        user_interest = user_hist_embed.mean(dim=1)

        mlp_input = torch.cat([user_interest, target_item_embed.squeeze(1)], dim=-1)
        output = self.mlp(mlp_input)
        return output

batch_size = 32
seq_length = 10
embed_dim = 16
hidden_dim = 64

user_hist = torch.randint(0, 1000, (batch_size, seq_length))
target_item = torch.randint(0, 1000, (batch_size,))
labels = torch.randint(0, 2, (batch_size,)).float()

dataset = TensorDataset(user_hist, target_item, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

din_model = DIN(embed_dim, hidden_dim, seq_length)
din_without_attention_model = DINWithoutAttention(embed_dim, hidden_dim, seq_length)

criterion = nn.BCELoss()
optimizer_din = optim.Adam(din_model.parameters(), lr=0.001)
optimizer_din_without_attention = optim.Adam(din_without_attention_model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    for user_hist, target_item, label in dataloader:
        optimizer_din.zero_grad()
        output = din_model(user_hist, target_item)
        loss = criterion(output.squeeze(), label)
        loss.backward()
        optimizer_din.step()

        optimizer_din_without_attention.zero_grad()
        output_without_attention = din_without_attention_model(user_hist, target_item)
        loss_without_attention = criterion(output_without_attention.squeeze(), label)
        loss_without_attention.backward()
        optimizer_din_without_attention.step()

    print(f"Epoch {epoch+1}/{num_epochs}, DIN Loss: {loss.item()}, DIN Without Attention Loss: {loss_without_attention.item()}")

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for user_hist, target_item, label in dataloader:
            output = model(user_hist, target_item)
            preds = output.squeeze().cpu().numpy()
            labels = label.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    auc = roc_auc_score(all_labels, all_preds)
    ctr = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return auc, ctr

din_auc, din_ctr = evaluate_model(din_model, dataloader)
print(f"DIN Model - AUC: {din_auc:.4f}, CTR: {din_ctr:.4f}")

din_without_attention_auc, din_without_attention_ctr = evaluate_model(din_without_attention_model, dataloader)
print(f"DIN Without Attention Model - AUC: {din_without_attention_auc:.4f}, CTR: {din_without_attention_ctr:.4f}")