import random
import numpy as np
import pandas as pd
from IPython.display import HTML
import time
import matplotlib.pyplot as plt


def generate_grid_randomly(n_rows, n_cols, n_holes=4):
    """
    Generate a random 2D Frozen Lake grid.

    The grid consists of:
    - 'F' : Frozen (safe) cells
    - 'H' : Holes (terminal failure states)
    - 'S' : Start state
    - 'G' : Goal state

    Args:
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        n_holes (int, optional): Number of hole cells to place.
            Defaults to 4.

    Returns:
        list[list[str]]: A 2D list representing the Frozen Lake grid.
    """
    
    grid = [["F" for _ in range(n_cols)] for _ in range(n_rows)]    
    states = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    
    random.shuffle(states)

    for i in range(0, n_holes):
        hole = states[i%(n_rows*n_cols)]
        r, c = hole
        grid[r][c] = "H"

    # start state
    r, c = states[-1]
    grid[r][c] = "S"
    
    # goal state
    r, c = states[-2]
    grid[r][c] = "G"

    return grid


class FrozenLakeEnvironment:
    def __init__(self, grid, reward_points, slippery=True):     
        self.grid = grid

        self.slippery = slippery
        self.action_to_idx = {
                               "left":  0,
                               "down": 1,
                               "right": 2,
                               "up": 3
                            }
        self.action_idx_to_name = {v:k for k,v in self.action_to_idx.items()}
        
        self.reward_points = reward_points
        
        self.terminal_states = ["G", "H"] 
        
        self.action_idx_to_step = {
             0: (0, -1), 
             1: (1, 0),
             2: (0, 1),
             3: (-1, 0)
        }

        self.all_action_idx = list(self.action_to_idx.values())
        self.n_actions = len(self.action_to_idx)
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols
        self._display_handle = None
    
    def move(self, current_state, action_idx):
        r, c = current_state
        # Note that we are representing state as a combination of row idx and column idx like a coordinate system 
        step_r, step_c = self.action_idx_to_step[action_idx]
        new_r = min(max(0, r+step_r), self.n_rows -1) #make sure step is a valid one
        new_c = min(max(0, c+step_c), self.n_cols -1)
        return new_r, new_c

    def is_terminal_state(self, state):
        r, c = state
        return self.grid[r][c] in self.terminal_states

        
    def step(self, current_state, action):
        """
        The core Model-Free interface. 
        Agent provides an action, environment returns (next_state, reward, done).
        """
        if self.is_terminal_state(current_state):
            return {"new_state": current_state,
                    "reward": 0,
                    "game_over": True}
                    
        actual_action = action
        if self.slippery:
            # 70% chance of success, 10% chance for each other direction
            roll = random.random()
            if roll < 0.7: # 70%
                actual_action = action
            elif roll < 0.8: # 10%
                actual_action = (action + 1) % self.n_actions
            elif roll < 0.9: # 10%
                actual_action = (action + 2) % self.n_actions
            else: # 10%
                actual_action = (action + 3) % self.n_actions
        else:
            actual_action = action
        
        new_state = self.move(current_state, actual_action)
        r, c = new_state
        cell = self.grid[r][c]
            
        if cell == "G":
            reward = 10
        elif cell == "H":
            reward = -10
        else:
            reward = -1
                
        game_over = cell in ["H", "G"]
        return {"new_state": new_state,
                "reward": reward,
                "game_over": game_over}


    def render(self, policy, current_state, episode_num=0, step_num=0,
               total_reward=0, sleep_time=0.0001):
            
            rows, cols = self.n_rows, self.n_cols
            icons = {"S": "🚀", "H": "🕳️", "G": "🏁", "F": "❄️"}
            arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        
            grid_display = []
            for r in range(rows):
                row_display = []
                for c in range(cols):
                    tile = self.grid[r][c]
                    if (r, c) == current_state:
                        content = "🤖" 
                    elif tile in ["S", "F"]:
                        action = policy[r][c]
                        content = f"{icons[tile]} {arrows[action]}"
                    else:
                        content = icons[tile]
                    row_display.append(content)
                grid_display.append(row_display)
        
            df = pd.DataFrame(grid_display)
        
            def style_cells(val):
                style = 'width: 60px; height: 60px; text-align: center; font-size: 20px; border: 1px solid #dee2e6;'
                if "🤖" in val: return style + 'background-color: #fff3cd; border: 2px solid #ffc107;' 
                if "🏁" in val: return style + 'background-color: #d4edda;' 
                if "🕳️" in val: return style + 'background-color: #f8d7da;' 
                return style + 'background-color: #89cfef;'
        
            # Generate the styled HTML for the table
            styled_html = df.style.map(style_cells).to_html()
            
            # Combine Header and Table into one HTML string
            header_html = f"<div style='font-family: sans-serif; margin-bottom: 10px;'>" \
                          f"<b>Episode:</b> {episode_num} | <b>Step:</b> {step_num} | <b>Score:</b> {total_reward}</div>"
            
            full_output = HTML(header_html + styled_html)
        
            if self._display_handle is None:
                # First time: display the combined HTML and capture handle
                self._display_handle = display(full_output, display_id=True)
            else:
                # Subsequent steps: update the same handle with the new combined HTML
                self._display_handle.update(full_output)
            
            time.sleep(sleep_time)