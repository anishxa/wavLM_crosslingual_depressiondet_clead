import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUDepressionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=2):
        super(GRUDepressionClassifier, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Self-attention pooling layer to aggregate over temporal sequence length
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, input_dim]
        mask: [batch_size, seq_len] (1 for valid steps, 0 for padded steps)
        """
        # GRU forward pass
        # out: [batch_size, seq_len, hidden_dim * 2]
        out, _ = self.gru(x)
        
        # Attention pooling
        # attn_logits: [batch_size, seq_len, 1]
        attn_logits = self.attention(out)
        if mask is not None:
            # Mask out padding steps by setting their attention weight logits to a large negative number
            mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_logits, dim=1)
        
        # pooled: [batch_size, hidden_dim * 2]
        pooled = torch.sum(out * attn_weights, dim=1)
        
        # Class logits
        logits = self.classifier(pooled)
        
        return logits
