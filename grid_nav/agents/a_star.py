import numpy as np
import heapq
from typing import List, Tuple, Optional

from grid_nav.agent import Agent


class AStarAgent(Agent):
    def _find_positions(self, state: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Find agent and target positions in the state."""
        agent_pos = None
        target_pos = None

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == 2:  # Agent
                    agent_pos = (i, j)
                elif state[i, j] == 3:  # Target
                    target_pos = (i, j)

        return agent_pos, target_pos

    def _get_neighbors(self, pos: Tuple[int, int], state: np.ndarray) -> List[Tuple[Tuple[int, int], str]]:
        """Get valid neighbors and the action to reach them."""
        neighbors = []
        row, col = pos

        # Check all four directions: Up, Down, Left, Right
        directions = [
            ((-1, 0), 'U'),  # Up
            ((1, 0), 'D'),   # Down
            ((0, -1), 'L'),  # Left
            ((0, 1), 'R')    # Right
        ]

        for (dr, dc), action in directions:
            new_row, new_col = row + dr, col + dc

            # Check if the new position is valid
            if (0 <= new_row < state.shape[0] and
                0 <= new_col < state.shape[1] and
                state[new_row, new_col] != 1):  # Not an obstacle
                neighbors.append(((new_row, new_col), action))

        return neighbors

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _a_star_search(self, start: Tuple[int, int], goal: Tuple[int, int], state: np.ndarray) -> List[str]:
        # Priority queue: (f_score, counter, position, path)
        # counter is used to break ties
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start, []))

        # Track visited positions
        closed_set = set()

        # g_score: cost from start to each position
        g_score = {start: 0}

        while open_set:
            _, _, current_pos, path = heapq.heappop(open_set)

            # Check if we reached the goal
            if current_pos == goal:
                return path

            # Skip if already visited
            if current_pos in closed_set:
                continue

            closed_set.add(current_pos)

            # Explore neighbors
            for neighbor_pos, action in self._get_neighbors(current_pos, state):
                if neighbor_pos in closed_set:
                    continue

                # Calculate new g_score (cost from start)
                tentative_g = g_score[current_pos] + 1

                # If we found a better path to this neighbor
                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    g_score[neighbor_pos] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor_pos, goal)

                    counter += 1
                    new_path = path + [action]
                    heapq.heappush(open_set, (f_score, counter, neighbor_pos, new_path))

        # No path found
        return []

    def get_cmd(self, state: np.ndarray) -> str:
        agent_pos, target_pos = self._find_positions(state)

        path = self._a_star_search(
            agent_pos,
            target_pos,
            state
        )

        if not path:
            return None  # No path found

        return path[0]
