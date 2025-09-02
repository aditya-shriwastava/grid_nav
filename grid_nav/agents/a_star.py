import numpy as np

from grid_nav.agent import Agent


class AStarAgent(Agent):
    def __init__(self):
        pass
        
    def get_cmd(self, state: np.ndarray) -> str:
        return 'L'
