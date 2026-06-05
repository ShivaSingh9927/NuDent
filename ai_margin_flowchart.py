"""Render a simple AI margin detection flowchart as a PNG."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_facecolor("#fafafa")

C_USER   = ("#cfe8ff", "#1f6feb")
C_ALGO   = ("#e6dafc", "#7b58c8")
C_DONE   = ("#d8f5d0", "#2ea043")

def box(x, y, w, h, title, subtitle, fill_edge):
    fill, edge = fill_edge
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        facecolor=fill, edgecolor=edge, linewidth=2,
    )
    ax.add_patch(p)
    ax.text(x, y + 1.6, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1d1d1f")
    ax.text(x, y - 1.8, subtitle, ha="center", va="center",
            fontsize=10, color="#3a3a3f")

def arrow(x1, y1, x2, y2, color="#444"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=2, color=color,
    )
    ax.add_patch(a)

# Title
ax.text(50, 97, "AI Margin Detection", ha="center", va="center",
        fontsize=18, fontweight="bold", color="#1d1d1f")
ax.text(50, 93.5, "How a single click becomes a closed margin loop",
        ha="center", va="center", fontsize=11, color="#6e6e73",
        style="italic")

# 1. Click
box(50, 86, 60, 7,
    "1.  Click on the prep tooth",
    "Drop a seed point anywhere on the cap",
    C_USER)
arrow(50, 82.5, 50, 78.5)

# 2. Measure curvature
box(50, 75, 60, 7,
    "2.  Measure surface curvature",
    "High-curvature ridges = candidate margin",
    C_ALGO)
arrow(50, 71.5, 50, 67.5)

# 3. Roll the ball
box(50, 64, 60, 7,
    "3.  Roll a virtual ball downhill",
    "Stops the moment it touches the ridge",
    C_ALGO)
arrow(50, 60.5, 50, 56.5)

# 4. Grow the loop
box(50, 53, 60, 7,
    "4.  Grow the loop along the ridge",
    "Flood-fill the connected ridge band",
    C_ALGO)
arrow(50, 49.5, 50, 45.5)

# 5. Smooth & close
box(50, 42, 60, 7,
    "5.  Order, smooth, close",
    "Wrap the band into one tidy loop",
    C_ALGO)
arrow(50, 38.5, 50, 34.5)

# 6. Done
box(50, 31, 60, 7,
    "Done  —  margin marked",
    "Add more seeds to fix any gaps  ·  Ctrl+Z to undo",
    C_DONE)

# Side note
ax.text(50, 22,
        "Extra clicks  →  re-runs only steps 4 & 5  (fast)",
        ha="center", va="center", fontsize=10.5,
        color="#7b58c8", style="italic")

# Mini legend
ax.add_patch(FancyBboxPatch((6, 6), 18, 6.5,
    boxstyle="round,pad=0.3,rounding_size=0.6",
    facecolor=C_USER[0], edgecolor=C_USER[1], linewidth=1.5))
ax.text(15, 9.2, "what you do", ha="center", va="center",
        fontsize=10, color="#1d1d1f")

ax.add_patch(FancyBboxPatch((28, 6), 18, 6.5,
    boxstyle="round,pad=0.3,rounding_size=0.6",
    facecolor=C_ALGO[0], edgecolor=C_ALGO[1], linewidth=1.5))
ax.text(37, 9.2, "what the AI does", ha="center", va="center",
        fontsize=10, color="#1d1d1f")

ax.add_patch(FancyBboxPatch((50, 6), 14, 6.5,
    boxstyle="round,pad=0.3,rounding_size=0.6",
    facecolor=C_DONE[0], edgecolor=C_DONE[1], linewidth=1.5))
ax.text(57, 9.2, "result", ha="center", va="center",
        fontsize=10, color="#1d1d1f")

plt.tight_layout()
out = "/home/shiva/Documents/NuDent/ai_margin_flowchart.png"
plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="#fafafa")
print(f"Wrote {out}")
