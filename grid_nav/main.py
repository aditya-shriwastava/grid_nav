#!/usr/bin/env python3

import argparse
import os
import sys
import warnings

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Suppress pkg_resources deprecation warning from pygame
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import pygame
import numpy as np
import random
import json
import datetime
import time
from grid_nav.agents.human import HumanAgent
from grid_nav.agents.a_star import AStarAgent
from grid_nav.agents.bc_agent import BCAgent


class GridNavGame:
    def __init__(self, world_file, agent, world_name, agent_name, headless=False, record=False, delay_action=False):
        self.cell_size = 40
        self.grid_size = 16
        self.status_height = 50
        self.window_width = self.cell_size * self.grid_size
        self.window_height = self.window_width + self.status_height
        
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        self.world = self._load_world(world_file)
        
        self.agent = agent
        self.world_name = world_name
        self.agent_name = agent_name
        self.headless = headless
        self.record = record
        self.delay_action = delay_action
        
        self.agent_pos = None
        self.target_pos = None
        self.initial_agent_pos = None
        self.game_over = False
        self.success = False
        self.game_over_time = None
        self.move_count = 0
        self.action_sequence = []
        self.start_time = datetime.datetime.now()
        
        self._init_pygame()
        self._place_agent_and_target()
        self._print_game_info()
    
    def _load_world(self, world_file):
        with open(world_file, 'r') as f:
            lines = f.readlines()
        
        world = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i, line in enumerate(lines[:self.grid_size]):
            for j, char in enumerate(line.strip()[:self.grid_size]):
                world[i, j] = int(char)
        return world
    
    def _init_pygame(self):
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Grid Navigation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.status_font = pygame.font.Font(None, 28)
    
    def _place_agent_and_target(self):
        free_spaces = [(i, j) for i in range(self.grid_size) 
                       for j in range(self.grid_size) if self.world[i, j] == 0]
        
        if len(free_spaces) < 2:
            raise ValueError("Not enough free spaces for agent and target")
        
        positions = random.sample(free_spaces, 2)
        self.agent_pos = positions[0]
        self.initial_agent_pos = positions[0]
        self.target_pos = positions[1]
    
    def _print_game_info(self):
        print(f"World: {self.world_name}")
        print(f"Agent: {self.agent_name}")
        print(f"Agent starting position: {self.agent_pos}")
        print(f"Target position: {self.target_pos}")
        print("-" * 40)
    
    def get_state(self):
        state = self.world.copy()
        if self.agent_pos:
            state[self.agent_pos[0], self.agent_pos[1]] = 2
        if self.target_pos:
            state[self.target_pos[0], self.target_pos[1]] = 3
        return state
    
    def move_agent(self, cmd):
        if self.game_over or not self.agent_pos or not cmd:
            return
        
        self.action_sequence.append(cmd)
        print(f"Action {self.move_count + 1}: {cmd}")
        
        row, col = self.agent_pos
        move_map = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        
        if cmd not in move_map:
            return
        
        dr, dc = move_map[cmd]
        new_row, new_col = row + dr, col + dc
        
        if self._is_out_of_bounds(new_row, new_col):
            self._end_game("GAME OVER: Attempted to move out of bounds!")
            return
        
        if self.world[new_row, new_col] == 1:
            self._end_game("GAME OVER: Hit an obstacle!")
            return
        
        self.agent_pos = (new_row, new_col)
        self.move_count += 1
        
        if self.delay_action:
            time.sleep(0.2)  # 200ms delay
        
        if self.agent_pos == self.target_pos:
            self.success = True
            self._end_game("SUCCESS: Reached target!")
    
    def _is_out_of_bounds(self, row, col):
        return row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size
    
    def _end_game(self, message):
        self.game_over = True
        if not self.headless:
            self.game_over_time = pygame.time.get_ticks()
        print(message)
        self._print_final_stats()
    
    def _print_final_stats(self):
        print("-" * 40)
        print(f"Total moves: {self.move_count}")
        print(f"Action sequence: {' -> '.join(self.action_sequence)}")
        print("-" * 40)
        
        if self.record:
            self._save_game_data()
    
    def _save_game_data(self):
        os.makedirs('data', exist_ok=True)
        
        file_timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S-%f")
        readable_timestamp = self.start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        game_data = {
            "grid": self.world_name,
            "agent_type": self.agent_name,
            "agent_start_coordinates": list(self.initial_agent_pos) if self.initial_agent_pos else None,
            "target_coordinates": list(self.target_pos) if self.target_pos else None,
            "moves": self.action_sequence,
            "total_moves": self.move_count,
            "success": self.success,
            "timestamp": readable_timestamp
        }
        
        filename = f"data/{file_timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)
        
        print(f"Game data recorded to: {filename}")
    
    def _draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x, y = j * self.cell_size, i * self.cell_size
                
                if self.world[i, j] == 1:
                    pygame.draw.rect(self.screen, self.BLACK, (x, y, self.cell_size, self.cell_size))
                else:
                    pygame.draw.rect(self.screen, self.WHITE, (x, y, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, self.GRAY, (x, y, self.cell_size, self.cell_size), 1)
    
    def _draw_entities(self):
        if self.target_pos:
            x = self.target_pos[1] * self.cell_size + self.cell_size // 2
            y = self.target_pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.GREEN, (x, y), self.cell_size // 3)
        
        if self.agent_pos:
            x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
            y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.BLUE, (x, y), self.cell_size // 3)
    
    def _draw_status_bar(self):
        pygame.draw.rect(self.screen, self.GRAY, (0, self.window_width, self.window_width, self.status_height))
        pygame.draw.line(self.screen, self.BLACK, (0, self.window_width), (self.window_width, self.window_width), 2)
        
        status_text = f"Moves: {self.move_count}"
        text = self.status_font.render(status_text, True, self.BLACK)
        text_rect = text.get_rect(center=(self.window_width // 2, self.window_width + self.status_height // 2))
        self.screen.blit(text, text_rect)
    
    def _draw_game_over_message(self):
        if self.game_over:
            message = "SUCCESS!" if self.success else "GAME OVER"
            color = self.GREEN if self.success else self.RED
            text = self.font.render(message, True, color)
            text_rect = text.get_rect(center=(self.window_width // 2, self.window_width // 2))
            self.screen.blit(text, text_rect)
    
    def draw(self):
        if self.headless:
            return
        
        self.screen.fill(self.WHITE)
        self._draw_grid()
        self._draw_entities()
        self._draw_status_bar()
        self._draw_game_over_message()
        pygame.display.flip()
    
    def run(self):
        running = True
        
        if self.headless:
            while running and not self.game_over:
                cmd = self.agent.get_cmd(self.get_state())
                if cmd is None:
                    running = False
                else:
                    self.move_agent(cmd)
        else:
            while running:
                self.draw()
                
                if self.game_over and self.game_over_time:
                    if pygame.time.get_ticks() - self.game_over_time >= 1000:
                        running = False
                        continue
                
                if not self.game_over:
                    cmd = self.agent.get_cmd(self.get_state())
                    if cmd is None:
                        running = False
                    else:
                        self.move_agent(cmd)
            
            pygame.quit()


def create_agent(agent_type, model_path=None):
    if agent_type == 'human':
        return HumanAgent()
    elif agent_type == 'A*':
        return AStarAgent()
    elif agent_type == 'bc':
        if model_path is None:
            print("Error: BC agent requires --model-path argument")
            return None
        return BCAgent(model_path)
    else:
        print(f"Agent type '{agent_type}' not implemented yet")
        return None


def validate_world_directory(world_directory):
    world_dir_path = f'worlds/{world_directory}'
    
    if not os.path.exists(world_dir_path):
        print(f"Error: World directory '{world_dir_path}' not found")
        return None
    
    try:
        grid_files = [f for f in os.listdir(world_dir_path) if f.endswith('.txt')]
        if not grid_files:
            print(f"Error: No grid files found in '{world_dir_path}'")
            return None
        return grid_files
    except OSError as e:
        print(f"Error accessing directory '{world_dir_path}': {e}")
        return None


def validate_specific_grid(world_directory, grid_name):
    world_file = f'worlds/{world_directory}/{grid_name}.txt'
    if not os.path.exists(world_file):
        print(f"Error: World file '{world_file}' not found")
        return False
    return True


def run_multiple_attempts(args, agent, world_directory, grid_files=None, specific_grid=None):
    successes = 0
    failures = 0
    
    for attempt in range(1, args.attempts + 1):
        print(f"\n=== ATTEMPT {attempt}/{args.attempts} ===")
        
        if grid_files:
            selected_file = random.choice(grid_files)
            grid_name = selected_file[:-4]
        else:
            grid_name = specific_grid
        
        world_file = f'worlds/{world_directory}/{grid_name}.txt'
        world_display_name = f"{world_directory}/{grid_name}"
        
        game = GridNavGame(world_file, agent, world_display_name, args.agent, 
                          headless=args.headless, record=args.record, delay_action=args.delay_action)
        game.run()
        
        if game.success:
            successes += 1
            print(f"SUCCESS! Attempt {attempt} completed successfully.")
        else:
            failures += 1
            print(f"FAILED! Attempt {attempt} was unsuccessful.")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total attempts: {args.attempts}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Success rate: {successes}/{args.attempts} ({100 * successes / args.attempts:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Grid Navigation')
    parser.add_argument('--world', nargs='+', default=['train'], 
                        metavar=('DIRECTORY', '[GRID]'), 
                        help='World directory and optional grid name')
    parser.add_argument('--agent', choices=['human', 'A*', 'bc'], default='human',
                        help='Agent type to use (default: human)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model file (required for bc agent)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without pygame display')
    parser.add_argument('--record', action='store_true',
                        help='Record game data to JSON file')
    parser.add_argument('--attempts', type=int, default=1,
                        help='Number of game attempts (default: 1)')
    parser.add_argument('--delay-action', action='store_true',
                        help='Add 200ms delay after each action')
    
    args = parser.parse_args()
    
    agent = create_agent(args.agent, args.model_path)
    if not agent:
        return
    
    if args.headless and args.agent == 'human':
        print("Error: Cannot use human agent in headless mode")
        return
    
    world_directory = args.world[0]
    
    if len(args.world) == 1:
        grid_files = validate_world_directory(world_directory)
        if not grid_files:
            return
        run_multiple_attempts(args, agent, world_directory, grid_files=grid_files)
    else:
        grid_name = args.world[1]
        if not validate_specific_grid(world_directory, grid_name):
            return
        run_multiple_attempts(args, agent, world_directory, specific_grid=grid_name)


if __name__ == '__main__':
    main()