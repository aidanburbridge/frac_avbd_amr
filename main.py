# MAIN

# Constants

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from solver import Solver
from bodies import rect_2D

def make_world():
    solver = Solver(dt=1/120, num_iterations=100, gravity=-9.81)
    solver.mu = 0.2       # friction
    solver.post_stabilize = True
    # Bodies
    
    ground = rect_2D(position=[0.0, -1.0, 0.3], velocity=[0,0,0], density=1.0, stiffness=1e6,
                     size=[20.0, 0.5], static=True)
    box1 = rect_2D(position=[0.0, 1.5, 0.5], velocity=[0,0,0], density=8.0, stiffness=1e5,
                   size=[0.5, 0.5], static=False)
    box2 = rect_2D(position=[ 0.0, 2.6, -0.1], velocity=[0,0,0], density=8.0, stiffness=1e5,
                   size=[0.6, 0.6], static=False)
    box3 = rect_2D(position=[ 0.0, 3.8,  0.3], velocity=[0,0,0], density=8.0, stiffness=1e5,
                   size=[0.5, 0.8], static=True)

    # Order doesn’t matter
    for b in (ground, box1, box2, box3):
        solver.add_body(b)

    return solver, [ground, box1, box2, box3]

def animate(steps=120, substeps=1):
    solver, bodies = make_world()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 5)
    ax.set_title("AVBD boxes + ground")
    ax.grid(True, alpha=0.2)

    patches = []
    for b in bodies:
        poly = Polygon(b.get_corners(), closed=True, fill=False, linewidth=2)
        ax.add_patch(poly)
        patches.append(poly)

    plt.pause(0.01)  # kick UI

    for t in range(steps):
        for _ in range(substeps):
            solver.step()
        for poly, b in zip(patches, bodies):
            poly.set_xy(b.get_corners())
        plt.pause(0.001)

    # Keep window open after sim
    plt.show()

if __name__ == "__main__":
    animate()