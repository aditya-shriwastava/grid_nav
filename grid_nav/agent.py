from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    @abstractmethod
    def get_cmd(self, state: np.ndarray) -> str:
        """
        state:
            * (16, 16) numpy array of integers, represent state of each cell
            * Rows stacked from top to bottom
            * Meaning of each integer:
                * 0: Free space
                * 1: Obstacle
                * 2: Agent
                * 3: Target
        return:
            * Value from {'L', 'R', 'U', 'D'} representing action that agent has to take
            * L: Left
            * R: Right
            * U: Up
            * D: Down
        Note:
            * Game terminates if Agent reaches the target following obstacle free path
            * Hitting the obstacle also terminated the game
        """
        pass
