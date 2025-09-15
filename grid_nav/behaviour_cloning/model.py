import torch
import torch.nn as nn
import torch.nn.functional as F


class GridNavCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.action_map = {
            'L': 0,
            'R': 1,
            'U': 2,
            'D': 3
        }
        self.action_map_inv = {v: k for k, v in self.action_map.items()}
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Max Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128*4*4, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 4)
        
    def forward(self, state, action=None):
        """
        state: torch.Tensor of integer from {0, 1, 2, 3}
           shape (batch_size, grid_size, grid_size)
        action: list of chars from {L, R, U, D}
        """
        x = self.encode_state(state)
        # x: (batch_size, 4, 16, 16)

        x = F.relu(self.conv1(x))
        # x: (batch_size, 32, 16, 16)

        x = F.relu(self.conv2(x))
        # x: (batch_size, 64, 16, 16)

        x = self.pool(x)
        # x: (batch_size, 64, 8, 8)

        x = F.relu(self.conv3(x))
        # x: (batch_size, 128, 8, 8)

        x = self.pool(x)
        # x: (batch_size, 128, 4, 4)
        
        # Flatten
        x = x.view(x.size(0), -1)
        # x: (batch_size, 128*4*4)
        
        x = F.relu(self.fc1(x))
        # x: (batch_size, 256)

        x = self.dropout(x)
        # x: (batch_size, 256)

        logits = self.fc2(x)
        # x: (batch_size, 4)

        loss = None
        if action is not None:
            a = torch.tensor([self.action_map[ai] for ai in action], dtype=torch.long)
            loss = F.cross_entropy(logits, a)
        
        return logits, loss
    
    def predict_action(self, state):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(state)
            action_idx = torch.argmax(logits, dim=-1)
            action = [self.action_map_inv[idx.item()] for idx in action_idx]
        return action

    # One Hot Encoding
    def encode_state(self, state):
        # Create encoded tensor directly in PyTorch on the same device as input
        batch_size = state.shape[0]
        encoded_state = torch.zeros(
            (batch_size, 4, 16, 16),
            dtype=torch.float32,
            device=state.device
        )

        encoded_state[:,0,:,:] = (state == 0).float()
        encoded_state[:,1,:,:] = (state == 1).float()
        encoded_state[:,2,:,:] = (state == 2).float()
        encoded_state[:,3,:,:] = (state == 3).float()

        return encoded_state
