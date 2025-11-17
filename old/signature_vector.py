import numpy as np
import math
import matplotlib.pyplot as plt

# Modulation function f(i) = 1 / (2^i * i!)
def modulation_f(k):
    return 1.0 / (2.0**k * math.factorial(k))

# Build a 2D grid graph with wrap-around (torus) on [0,1]x[0,1]
def build_grid_neighbors(L):
    N = L * L
    neighbors = np.zeros((N, 4), dtype=int)
    d = np.full(N, 4.0)  # degree is always 4

    def idx(x, y):
        return x * L + y

    for x in range(L):
        for y in range(L):
            node = idx(x, y)
            right = idx(x, (y + 1) % L)
            left  = idx(x, (y - 1) % L)
            down  = idx((x + 1) % L, y)
            up    = idx((x - 1) % L, y)

            neighbors[node, 0] = right
            neighbors[node, 1] = left
            neighbors[node, 2] = down
            neighbors[node, 3] = up

    return neighbors, d

# Algorithm 1: construct random feature vector φ_f(i)
def random_feature_vector(i, neighbors, d, p_halt, m):
    N = neighbors.shape[0]
    L = int(round(N**0.5))                 # EDIT: infer grid side length n (=L)
    beta = (L * L) / 2.0                   # EDIT: resolution-aware scale β(n)=n^2/2

    phi = np.zeros(N)
    rng = np.random.default_rng(0)         # fixed seed

    for _ in range(m):
        load = 1.0
        current_node = i
        terminated = False
        walk_length = 0
        fk = 1.0                           # EDIT: f_n(0) = 1

        while not terminated:
            # line 8 (replaced): use running coefficient instead of factorial call
            phi[current_node] += load * fk  # EDIT: accumulate with fk instead of modulation_f

            walk_length += 1
            fk *= beta / walk_length        # EDIT: successive multiplication: fk <- β/k * fk

            # line 10: choose random neighbor
            neighs = neighbors[current_node]
            new_node = neighs[rng.integers(0, len(neighs))]

            # unweighted grid but using transition weight (your choice): W = 1/d
            W_entry = 1.0 / d[current_node]
            load = load * d[current_node] / (1.0 - p_halt) * W_entry
            current_node = new_node
            terminated = (rng.random() < p_halt)

    # line 16
    phi /= m
    return phi

# Fixed parameters across resolutions
p_halt = 0.1
m = 100000

grid_sizes = [10, 20, 40]
phis = []

for L in grid_sizes:
    neighbors, d = build_grid_neighbors(L)
    center_node = (L // 2) * L + (L // 2)
    phi = random_feature_vector(center_node, neighbors, d, p_halt, m)
    phis.append((L, phi.reshape(L, L)))

# Visualization of loads for increasing resolution
fig, axes = plt.subplots(1, len(grid_sizes), figsize=(15, 4))

for ax, (L, phi_grid) in zip(axes, phis):
    im = ax.imshow(phi_grid)   # default colormap
    ax.set_title(f"L = {L}")
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
plt.tight_layout()
plt.show()