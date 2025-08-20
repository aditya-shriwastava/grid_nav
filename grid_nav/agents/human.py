import pygame
import numpy as np

from grid_nav.agent import Agent


class HumanAgent(Agent):
    def __init__(self):
        self.waiting_for_input = True
        self.last_cmd = None
        
    def get_cmd(self, state: np.ndarray) -> str:
        # Block until a key is pressed
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None  # Quit game
                elif event.type == pygame.KEYDOWN:
                    # Check for movement keys
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        return 'U'
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        return 'D'
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        return 'L'
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        return 'R'
                    elif event.key == pygame.K_q:
                        return None  # Quit game
            
            # Small delay to prevent CPU spinning
            pygame.time.wait(10)
