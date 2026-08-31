"""Redraw the rollout figure from rollout3d.py --dump, with no title.

Same panels, same palette, same limits as rollout3d.py. The only differences:
the suptitle is gone, and the 3D box is zoomed in slightly inside each axes so
the tick labels have room instead of being clipped by the axes edge.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

SUB = ["Grab bolt (L)", "Place in hole (L)", "Grab driver (R)", "Screw in (R)", "Home"]
COL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

d = np.load("rollout3d.npz")
H, P, S = d["human"], d["policy"], d["subtask"]

fig = plt.figure(figsize=(15.4, 6.2))
for n, (el, az, ttl) in enumerate([(22, -60, "perspective"),
                                   (90, -90, "top-down (x-y)"),
                                   (0, -90, "side (x-z)")]):
    ax = fig.add_subplot(1, 3, n + 1, projection="3d")
    ax.plot(*H.T, color="0.25", lw=2.4, label="human (ground truth)", zorder=3)
    for s in sorted(set(S.tolist())):
        k = S == s
        ax.plot(*P[k].T, color=COL[s], lw=2.0, label=f"policy · {SUB[s]}", zorder=4)
    for i in range(0, len(P), max(1, len(P) // 45)):
        ax.plot(*np.stack([H[i], P[i]]).T, color="#C44E52", lw=0.7, alpha=0.55, zorder=2)
    ax.scatter(*H[0], c="k", s=45, marker="o", zorder=5)
    ax.scatter(*H[-1], c="k", s=55, marker="X", zorder=5)
    ax.view_init(elev=el, azim=az)
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("x (mm)", labelpad=10)
    ax.set_ylabel("y (mm)", labelpad=10 if n != 1 else 14)
    ax.set_zlabel("z (mm)", labelpad=10 if n != 2 else 18)
    if n == 1:
        ax.set_zticks([]); ax.set_zlabel("")
    elif n == 2:
        ax.set_yticks([]); ax.set_ylabel("")
    #zoom < 1 shrinks the drawn box inside the axes rectangle, which is what
    #gives the tick labels room; without it the outermost label is cut in half
    ax.set_box_aspect(None, zoom=0.84)
    allpts = np.vstack([H, P])
    c, r = allpts.mean(0), (allpts.max(0) - allpts.min(0)).max() / 2 + 10
    for setter, ci in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(c[ci] - r, c[ci] + r)
    if n == 0:
        ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("figs/rollout3d.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote figs/rollout3d.png")
