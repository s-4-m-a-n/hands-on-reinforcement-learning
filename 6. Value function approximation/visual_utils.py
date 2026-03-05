import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML
import pandas as pd
from IPython.display import display
from frozen_lake_environment import State
from matplotlib.collections import LineCollection
import matplotlib.offsetbox as moffsetbox
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_EMOJI_FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", 109
)

EMOJI_MAPS = {"S": "🚀", "H": "🕳️", "G": "🏁", "F": ""}
ARROWS_MAPS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

def _emoji_image(char: str, out_size: int = 80) -> np.ndarray:
    """Render a single emoji glyph to a centred RGBA numpy array."""
    N = 109
    img  = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=_EMOJI_FONT, embedded_color=True)
    x = (N - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (N - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), char, font=_EMOJI_FONT, embedded_color=True)
    return np.asarray(img.resize((out_size, out_size), Image.LANCZOS))


def _place_emoji(ax, char: str, x: float, y: float,
                 zoom: float = 0.55, zorder: int = 10):
    """Overlay an emoji at data-coordinate (x, y) on the given axes."""
    arr = _emoji_image(char)
    im  = moffsetbox.OffsetImage(arr, zoom=zoom)
    ab  = moffsetbox.AnnotationBbox(im, (x, y), frameon=False, zorder=zorder)
    ax.add_artist(ab)


def plot_trajectory_history(env, trajs, policy=None, jitter=0.03, alpha=0.05):
    rows, cols = env.n_rows, env.n_cols
    fig, ax = plt.subplots(figsize=(4, 4))

    # ── 1. Background grid ────────────────────────────────────────────────────
    tile_colors = {
        "S": "#cce5ff",   # blue  – start
        "H": "#f8d7da",   # red   – hole
        "G": "#d4edda",   # green – goal
        "F": "#89cfef",   # light blue – frozen
    }
    ARROWS_MAPS = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    for r in range(rows):
        for c in range(cols):
            tile = env.grid[r][c]

            # Coloured cell
            rect = plt.Rectangle(
                (c, rows - r - 1), 1, 1,
                facecolor=tile_colors[tile],
                edgecolor="white", lw=1.5, zorder=0,
            )
            ax.add_patch(rect)

            # Policy arrow (plain text is fine — no emoji font needed)
            if policy is not None:
                if tile in ("S", "F"):
                    state_idx = r * cols + c
                    action    = policy[state_idx]
                    ax.text(
                        c + 0.8, rows - r - 0.2,
                        ARROWS_MAPS[action],
                        color="green", fontsize=12,
                        ha="center", va="center", zorder=5,
                    )

    # ── 2. Hand-drawn trajectory strokes ─────────────────────────────────────
    all_strokes = []
    for path in trajs:
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            x1, y1 = c1 + 0.5, rows - r1 - 0.5
            x2, y2 = c2 + 0.5, rows - r2 - 0.5
            # Slight random wobble on the midpoint for a hand-drawn feel
            mx = (x1 + x2) / 2 + np.random.uniform(-jitter, jitter)
            my = (y1 + y2) / 2 + np.random.uniform(-jitter, jitter)
            all_strokes.append(np.array([[x1, y1], [mx, my], [x2, y2]]))

    lc = LineCollection(
        all_strokes,
        linewidths=0.5,
        colors="red",
        alpha=alpha,
        capstyle="round",
        zorder=1,
    )
    ax.add_collection(lc)

    # ── 3. Emoji icons (rendered via Pillow, overlaid as images) ─────────────

    for r in range(rows):
        for c in range(cols):
            tile = env.grid[r][c]
            if tile in EMOJI_MAPS:
                x, y = c + 0.5, rows - r - 0.5
                _place_emoji(ax, EMOJI_MAPS[tile], x, y, zoom=0.2, zorder=10)

    # ── Final display settings ────────────────────────────────────────────────
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    

def render_policy_and_value(env, policy, Q=None):
    """
    Render the policy and optionally the state-value function in a 
    visually appealing grid format using emojis and arrows.

    Args:
        env: Frozen Lake environment object with `grid`, `n_rows`, `n_cols`
        policy: 2D array of actions for each state
        Q: 2D array of state-action values (optional)
    """
    rows, cols = env.n_rows, env.n_cols
    
    # --- Policy Display ---
    grid_policy = []
    for r in range(rows):
        row_display = []
        for c in range(cols):
            tile = env.grid[r][c]
            if tile in ["S", "F"]:
                state_idx = State((r, c), env.n_cols).idx
                action = policy[state_idx]
                content = f"{EMOJI_MAPS[tile]} {ARROWS_MAPS[action]}"
            else:
                content = EMOJI_MAPS[tile]
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
        print("State-Value Function:")
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
                ax.text(c, r, ARROWS_MAPS[action], ha='center', va='center', fontsize=15, color='red')

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


