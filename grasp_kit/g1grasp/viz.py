"""A picture to check a result by eye: the image with the target's mask and the chosen hand pose, and the target's
points from above and from the side with the table's height, the grasp and the pregrasp."""

import numpy as np

from .frames import project


def check_image(path, rgb, K, T_base_cam, mask, cloud, table_top, chosen=None, alternatives=(), title="") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    img = rgb.astype(float) / 255
    if mask is not None:
        img[mask] = 0.55 * img[mask] + 0.45 * np.array([0.1, 0.9, 0.2])
    ax[0].imshow(img)
    ax[0].set_axis_off()

    def hand(T, colour, lw, label=None):
        p, x, y = T[:3, 3], T[:3, 0], T[:3, 1]
        uv = project(T_base_cam, K, [p, p + 0.08 * x, p + 0.05 * y])
        ax[0].plot(*uv[[0, 1]].T, color=colour, lw=lw, label=label)
        ax[0].plot(*uv[[0, 2]].T, color=colour, lw=lw, ls=":")
        ax[0].plot(*uv[0], "o", color=colour, ms=3 + 2 * lw)
        tip = p + 0.08 * x
        for a, j in ((ax[1], 1), (ax[2], 2)):  # from above (x, y), from the side (x, z)
            a.plot([p[0], tip[0]], [p[j], tip[j]], color=colour, lw=lw)
            a.plot(p[0], p[j], "o", color=colour, ms=3 + 2 * lw)

    for T in alternatives:
        hand(np.asarray(T), "0.6", 1)
    if chosen is not None:
        hand(np.asarray(chosen["palm"]), "red", 2.5, "grasp (palm link; dotted: palm normal)")
        hand(np.asarray(chosen["pregrasp_palm"]), "royalblue", 2, "pregrasp")
        ax[0].legend(loc="lower left", fontsize=8)
    if cloud is not None and len(cloud):
        ax[1].scatter(cloud[:, 0], cloud[:, 1], s=1, c="green")
        ax[2].scatter(cloud[:, 0], cloud[:, 2], s=1, c="green")
        ax[2].axhline(table_top, color="0.4", ls="--", lw=1)
        ax[2].text(cloud[:, 0].min(), table_top, f" table {table_top:+.3f} m", va="bottom", fontsize=8, color="0.3")
    ax[1].set(title="from above (base frame)", xlabel="x forward (m)", ylabel="y left (m)", aspect="equal")
    ax[2].set(title="from the side", xlabel="x forward (m)", ylabel="z up (m)", aspect="equal")
    for a in ax[1:]:
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
