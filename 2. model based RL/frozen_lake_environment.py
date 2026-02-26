import random
import numpy as np

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

        
    def get_transition_prob(self, state, action):
        if self.is_terminal_state(state):
            # if it is terminal state do not move
            return [{"prob": 1.0,
                     "new_state": state,
                     "reward": 0,
                     "done": True}]

        # --- deterministic action -------
        actions_prob = {
            action: 1.0
        }
        
        # --- stochastic action ----------
        if self.slippery: 
            # action-prob pair
            actions_prob = {
                action: 0.7,
                (action + 1)%self.n_actions: 0.1,
                (action + 2)%self.n_actions: 0.1,
                (action + 3)%self.n_actions: 0.1
            }
        # --------------------------------
        transitions = []
        for action, prob in actions_prob.items():
            new_state = self.move(state, action)
            r, c = new_state
            cell = self.grid[r][c]
                       
            reward = self.reward_points[cell]    
            
            game_over = cell in self.terminal_states
            transitions.append({"prob": prob,
                                "new_state": new_state,
                                "reward": reward,
                                "done": game_over})
        return transitions