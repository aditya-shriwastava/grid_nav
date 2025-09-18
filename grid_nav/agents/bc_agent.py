import torch
import numpy as np
from pathlib import Path

from grid_nav.agent import Agent
from grid_nav.behaviour_cloning.model import GridNavCNN


class BCAgent(Agent):
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = GridNavCNN()

        self.model.load_state_dict(
            torch.load(self.model_path, map_location='cpu')['model_state_dict']
        )
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        print(f"BCAgent initialized with model: {model_path}")
        print(f"Using device: {self.device}")

    def get_cmd(self, state: np.ndarray) -> str:
        with torch.no_grad():
            state_batch = np.expand_dims(state, axis=0)
            state_tensor = torch.from_numpy(state_batch).to(self.device)

            logits, _ = self.model(state_tensor, action=None)

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample from the action distribution
            action_idx = torch.multinomial(probs[0], 1).item()
            action = self.model.action_map_inv[action_idx]

            return action
