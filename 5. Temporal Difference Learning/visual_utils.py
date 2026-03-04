import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML
import pandas as pd
from IPython.display import display
from frozen_lake_environment import State

def render_policy_and_value(env, policy, Q=None):
    """
    Render the policy and optionally the state-value function in a 
    visually appealing grid format using emojis and arrows.

    Args:
        env: Frozen Lake environment object with `grid`, `n_rows`, `n_cols`
        policy: 2D array of actions for each state
        Q: 2D array of state actions (optional)
    """
    rows, cols = env.n_rows, env.n_cols
    
    # Icons and arrows
    icons = {"S": "🚀", "H": "🕳️", "G": "🏁", "F": ""}
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    # --- Policy Display ---
    grid_policy = []
    for r in range(rows):
        row_display = []
        for c in range(cols):
            tile = env.grid[r][c]
            if tile in ["S", "F"]:
                state_idx = State((r, c), env.n_cols).idx
                action = policy[state_idx]
                content = f"{icons[tile]} {arrows[action]}"
            else:
                content = icons[tile]
            row_display.append(content)
        grid_policy.append(row_display)

    df_policy = pd.DataFrame(grid_policy)
    
    # --- state action Value Function Display ---
    if Q is not None:
        grid_value = []
        for state_idx in range(env.n_states):
            row_display = []
            for action_idx in range(env.n_actions):
                val = Q[state_idx][action_idx]
                row_display.append(f"{val:.2f}")
            grid_value.append(row_display)
        df_value = pd.DataFrame(grid_value)

    # --- Styling function ---
    def style_cells(val):
        style = 'width: 60px; height: 60px; text-align: center; font-size: 20px; border: 1px solid #dee2e6;'
        if "🏁" in val: return style + 'background-color: #d4edda;' # Green
        if "🕳️" in val: return style + 'background-color: #f8d7da;' # Red
        if "🚀" in val: return style + 'background-color: #cce5ff;' # Blue
        return style + 'background-color: #89cfef;'  # Frozen tiles

    # --- Render ---
    print("Policy:")
    display(df_policy.style.map(style_cells))
    
    if Q is not None:
        print("state-action value Function:")
        display(df_value.style.set_properties(**{
            'width': '60px', 
            'height': '60px', 
            'text-align': 'center', 
            'font-size': '16px',
            'border': '1px solid #dee2e6'
        }))


def animate_policy_value_video(env, policy_history, V_history=None, interval=500):
    """
    Animate the evolution of policy and state-value function like a video.

    Args:
        env: Frozen Lake environment with `grid`, `n_rows`, `n_cols`
        policy_history: List of 2D arrays of actions
        V_history: Optional list of 2D arrays of state values
        interval: Time between frames in milliseconds
    """
    rows, cols = env.n_rows, env.n_cols
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    fig, ax = plt.subplots(figsize=(cols//2, rows//2))
    
    def update(frame):
        ax.clear()
        ax.set_xticks(np.arange(cols+1)-0.5, minor=False)
        ax.set_yticks(np.arange(rows+1)-0.5, minor=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.invert_yaxis()

        policy = policy_history[frame]
        if V_history is not None:
            V = V_history[frame]
            # Plot heatmap
            im = ax.imshow(V, cmap='Blues', alpha=0.6)
            for (r, c), val in np.ndenumerate(V):
                ax.text(c, r, f"{val:.1f}", ha='center', va='center', fontsize=15)
        else:
            im = None

        # Overlay policy arrows
        for r in range(rows):
            for c in range(cols):
                tile = env.grid[r][c]
                if tile in ['H', 'G', 'S']:
                    continue
                state_idx = State((r, c), env.n_cols).idx
                action = policy[state_idx]
                ax.text(c, r, arrows[action], ha='center', va='center', fontsize=15, color='red')

        # Overlay start, goal, hole
        for r in range(rows):
            for c in range(cols):
                tile = env.grid[r][c]
                if tile == 'S':
                    ax.text(c, r, 'S', ha='center', va='center', fontsize=15)
                elif tile == 'G':
                    ax.text(c, r, 'G', ha='center', va='center', fontsize=15)
                elif tile == 'H':
                    ax.text(c, r, 'H', ha='center', va='center', fontsize=15)

        ax.set_title(f"Iteration {frame+1}")

    ani = animation.FuncAnimation(fig, update, frames=len(policy_history), interval=interval)
    plt.close(fig)  # Prevent double display in notebooks
    return ani


