import json
import os
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class GridNavDataset(Dataset):
    def __init__(self, trajectory_paths: List[str]):
        self.trajectory_paths = trajectory_paths

        # Storage for all state-action pairs
        self.states = []
        self.actions = []

        # Load all trajectories
        self._load_all_trajectories()

    def _load_all_trajectories(self):
        """Load all trajectories from the provided paths."""
        print(f"Loading {len(self.trajectory_paths)} trajectory files...")

        loaded_count = 0
        skipped_count = 0

        for traj_path in self.trajectory_paths:
            if not os.path.exists(traj_path):
                print(f"Warning: Trajectory file not found: {traj_path}")
                skipped_count += 1
                continue

            try:
                with open(traj_path, 'r') as f:
                    trajectory = json.load(f)

                # Process this trajectory
                states, actions = self._process_trajectory(trajectory)

                if states and actions:
                    self.states.extend(states)
                    self.actions.extend(actions)
                    loaded_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"Error loading {traj_path}: {e}")
                skipped_count += 1

    def _load_world(self, world_path: str) -> np.ndarray:
        """
        Load world grid from file.

        Args:
            world_path: Path to world file

        Returns:
            16x16 numpy array representing the world
        """
        with open(world_path, 'r') as f:
            lines = f.readlines()

        world = np.zeros((16, 16), dtype=int)
        for i, line in enumerate(lines[:16]):
            for j, char in enumerate(line.strip()[:16]):
                world[i, j] = int(char)
        return world

    def _process_trajectory(self, trajectory: Dict) -> Tuple[List[np.ndarray], List[int]]:
        """
        Process a single trajectory to extract state-action pairs.

        Args:
            trajectory: Dictionary containing trajectory data

        Returns:
            states: List of grid states
            actions: List of action indices
        """
        # Extract trajectory information
        grid_name = trajectory.get('grid')
        agent_start = trajectory.get('agent_start_coordinates')
        target_pos = trajectory.get('target_coordinates')
        moves = trajectory.get('moves', [])

        # Validate trajectory data
        if not all([grid_name, agent_start, target_pos, moves]):
            return [], []

        # Load the world
        world_path = os.path.join("worlds", f"{grid_name}.txt")
        try:
            world = self._load_world(world_path)
        except Exception as e:
            print(f"Error loading world {world_path}: {e}")
            return [], []

        # Generate states and actions
        states = []
        actions = []
        agent_pos = list(agent_start)

        for move in moves:
            # Create current state
            state = world.copy()
            state[agent_pos[0], agent_pos[1]] = 2  # Mark agent
            state[target_pos[0], target_pos[1]] = 3  # Mark target

            states.append(state)
            actions.append(move)

            # Update agent position for next state
            if move == 'U':
                agent_pos[0] -= 1
            elif move == 'D':
                agent_pos[0] += 1
            elif move == 'L':
                agent_pos[1] -= 1
            elif move == 'R':
                agent_pos[1] += 1

        return states, actions

    def __len__(self) -> int:
        """Return the number of state-action pairs."""
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get a single state-action pair.

        Args:
            idx: Index of the sample

        Returns:
            state: Grid state as numpy array (16, 16)
            action: Action index (0-3)
        """
        return self.states[idx], self.actions[idx]
