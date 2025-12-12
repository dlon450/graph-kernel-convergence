from manifolds import ManifoldConfig, Sphere, Ellipsoid
from plots import (
    plot_dynamics,
    plot_pred_vs_actual,
    plot_error_vs_truth,
    visualize_several_validation_starts,
    plot_kernel_row_for_start,
)
import os

filepath = "saved/sphere_manifold.pkl" # loads manifold from filepath if it exists
os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Define manifold configuration and build manifold
cfg = ManifoldConfig(N=4000, k=8, num_starts=1000, n_walks=10000, phalt=0.01, t=20.0)
M = Sphere(cfg) # Ellipsoid(cfg, a=1.0, b=1.3, c=0.7)
if os.path.exists(filepath):
    M.load_pickle(filepath)
    model, hist, splits = M.model, M.history, M.splits
else:
    M.build_signature_walks()
    # M.make_dataset()
# model, hist, splits = M.train_model()
# M.save_pickle("saved/sphere_manifold_MSE.pkl")
K_nn, K_true, G_nn, mse, rel = M.compute_ground_truth(L_max=50, t=0.025)

# # Collect validation predictions
# y_true_va, y_pred_va = M.collect_val_predictions()
# plot_dynamics(hist, title="Sphere: RMSE", rmse=True, savepath="saved/manifolds/sphere/sphere_rmse.pdf")
# plot_dynamics(hist, title="Sphere: Relative error", rmse=False, savepath="saved/manifolds/sphere/sphere_rel_error.pdf")
# plot_pred_vs_actual(y_true_va, y_pred_va, savepath="saved/manifolds/sphere/sphere_pred_vs_actual.png")
# plot_error_vs_truth(y_true_va, y_pred_va, error_type="relative", savepath="saved/manifolds/sphere/sphere_error_vs_truth_relative.png")
# plot_error_vs_truth(y_true_va, y_pred_va, error_type="squared", savepath="saved/manifolds/sphere/sphere_error_vs_truth_squared.png")

# # f(x, ω) on the manifold for a few validation starts
# visualize_several_validation_starts(model,
#     M.X,                 # N x 3
#     M.start_indices,     # num_starts
#     M.phi,               # num_starts x N
#     val_start_ids=splits["val_starts"],
#     how_many=3,
#     geod_rows=None,           # we recompute geodesic via dot product on S²
#     title_prefix="Sphere | val",
# )

# Kernel row plots (NN vs ground truth heat kernel)
example_start = int(splits["val_starts"][0])
plot_kernel_row_for_start(
    M.X,
    K_true,
    K_nn,
    start_idx=example_start,
    title_prefix="Heat kernel",
    savepath="saved/manifolds/sphere/sphere_heat_kernel_comparison.png",
)