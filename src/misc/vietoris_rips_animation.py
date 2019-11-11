import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix
from src.utils.persistent_diagram.vietoris_rips import vietoris_rips_persistence_diagram


# Examples:
# python -m src.misc.vietoris_rips_animation -n 20 --shape random --noise 0
# python -m src.misc.vietoris_rips_animation -n 50 --shape circle --noise 0
# python -m src.misc.vietoris_rips_animation -n 50 --shape circle --noise 0.05
# python -m src.misc.vietoris_rips_animation -n 50 --shape 2-circle --noise 0.05


parser = argparse.ArgumentParser(description='Animation for Vietoris-Rips filtration')
parser.add_argument("-n", type=int, default=20, help="Number of points")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--shape", type=str, default='random')
parser.add_argument("--noise", type=float, default=0.05, help="Noise scale")
parser.add_argument("--max_dist", type=float, default=2., help="Maximum distance for the filtration")

COLORS = ['black', 'red']


def generate_points(num_pts):
    pts = np.random.rand(num_pts, 2)
    return pts


def generate_circle(x, y, r, num_pts, noise=0.):
    theta = np.random.rand(num_pts) * 2 * np.pi
    pts = np.zeros([num_pts, 2])
    pts[:, 0] = np.sin(theta) * r + x
    pts[:, 1] = np.cos(theta) * r + y
    return pts + np.random.normal(0., 1., [num_pts, 2]) * noise


def plot_point_cloud(ax, pts, dist, max_dist):
    ax.set_aspect('equal')
    ax.scatter(pts[:, 0], pts[:, 1], color='black', s=5)

    circles = []
    for i in range(len(pts)):
        c = plt.Circle(pts[i], 0, alpha=0.2)
        circles.append(c)
        ax.add_artist(c)

    segments = [[None for _ in range(len(pts))] for _ in range(len(pts))]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if dist[i][j] < max_dist:
                s = plt.Line2D([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]])
                s.set_visible(False)
                segments[i][j] = segments[j][i] = s
                ax.add_artist(s)
    return circles, segments


def plot_persistence_diagrams(ax, pds, max_dist):
    ax.set_aspect('equal')
    ax.set_xlim(0, max_dist / 2 + 0.1)
    ax.set_ylim(0, max_dist / 2 + 0.1)
    ax.plot([0, max_dist], [0, max_dist], color='black', linewidth=1)
    ax.set_title('Persistence Diagram')
    for i in range(2):
        pairs = pds[i].to_list()
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=8, color=COLORS[i])
    timeline_horz = plt.Line2D([0, 0], [0, max_dist], color='green', linewidth=1)
    timeline_vert = plt.Line2D([0, max_dist], [0, 0], color='green', linewidth=1)
    ax.add_artist(timeline_horz)
    ax.add_artist(timeline_vert)
    return timeline_horz, timeline_vert


def plot_persistence_barcodes(ax, pds, max_dist):
    ax.set_xlim(0, max_dist / 2)
    pb_timeline = plt.Line2D([0, 1], [0, 0], color='green', linewidth=1)
    ax.add_artist(pb_timeline)
    ax.set_title('Persistence Barcode')
    ax.set_ylim(0, len(pds[0]) + len(pds[1]) + 1)
    # plot barcode
    cnt = 1
    for i in range(2):
        pairs = pds[i].to_list()
        for birth, death in pairs:
            ax.plot([birth, death if death != float('inf') else max_dist / 2], [cnt, cnt], color=COLORS[i])
            cnt += 1
    return pb_timeline


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    num_pts = args.n
    
    if args.shape == 'random':
        pts = generate_points(num_pts)
    elif args.shape == 'circle':
        pts = generate_circle(0, 0, 0.5, num_pts, noise=args.noise)
    elif args.shape == '2-circle':
        pts = np.concatenate([
            generate_circle(-1, 0, 0.5, num_pts // 2, noise=args.noise),
            generate_circle(1, 0, 0.5, num_pts // 2, noise=args.noise)
        ], 0)
    else:
        raise ValueError

    pds = vietoris_rips_persistence_diagram(pts, args.max_dist)

    dist = distance_matrix(pts, pts)

    fig = plt.figure()
    gridspec = fig.add_gridspec(
        ncols=2, nrows=2,
        width_ratios=[1, 1], height_ratios=[2, 1])
    ax_pc = plt.subplot(gridspec[0, 0])
    ax_pd = plt.subplot(gridspec[0, 1])
    ax_pb = plt.subplot(gridspec[1, :])

    circles, segments = plot_point_cloud(ax_pc, pts, dist, args.max_dist)
    timeline_horz, timeline_vert = plot_persistence_diagrams(ax_pd, pds, args.max_dist)
    pb_timeline = plot_persistence_barcodes(ax_pb, pds, args.max_dist)

    # Run animation
    anim_running = False

    def on_click(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    def update(frame):
        for c in circles:
            c.set_radius(frame / 2)
        for i in range(num_pts):
            for j in range(i + 1, num_pts):
                if segments[i][j] is not None:
                    segments[i][j].set_visible(dist[i][j] <= frame)
        timeline_vert.set_data([frame, frame], [0, args.max_dist])
        timeline_horz.set_data([0, args.max_dist], [frame, frame])
        pb_timeline.set_data([frame, frame], [0, len(pds[0]) + len(pds[1]) + 1])
        return circles, timeline_horz, timeline_vert

    fig.canvas.mpl_connect('button_press_event', on_click)

    anim = FuncAnimation(
        fig, update, frames=np.linspace(0, args.max_dist / 2, 250),
        interval=50,
        repeat_delay=500,
        blit=False)
    anim.event_source.stop()
    plt.show()


main()
