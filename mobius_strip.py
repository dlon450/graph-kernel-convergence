from manifolds import ManifoldConfig, MobiusStrip
from plots import (
    plot_dynamics,
    plot_pred_vs_actual,
    plot_error_vs_truth,
    visualize_several_validation_starts,
    plot_kernel_row_for_start,
)
import os

filepath = "saved/mobius_strip_manifold.pkl" # loads manifold from filepath if it exists
os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Define manifold configuration and build manifold
cfg = ManifoldConfig(N=4000, k=8, num_starts=1000, n_walks=10000, phalt=0.01, t=20.0, epochs=1000)
M = MobiusStrip(cfg)
if os.path.exists(filepath):
    M.load_pickle(filepath)
    model, hist, splits = M.model, M.history, M.splits
else:
    M.build_signature_walks()
    M.make_dataset()
    model, hist, splits = M.train_model()
    M.save_pickle(filepath)

# Collect validation predictions
y_true_va, y_pred_va = M.collect_val_predictions()
plot_dynamics(hist, title="Mobius Strip: RMSE", rmse=True)
plot_dynamics(hist, title="Mobius Strip: Relative error", rmse=False)
plot_pred_vs_actual(y_true_va, y_pred_va, savepath="mobius_strip_pred_vs_actual.pdf")
plot_error_vs_truth(y_true_va, y_pred_va, error_type="relative")
plot_error_vs_truth(y_true_va, y_pred_va, error_type="squared")

# f(x, ω) on the manifold for a few validation starts
geod_matrix = M._geodesic_matrix_for_training()
geod_rows = {
    int(s): geod_matrix[int(s)]             # row j corresponds to start_indices[j] == s
    for s in M.start_indices
}
visualize_several_validation_starts(
    model,
    M.X,                 # N x 3
    M.start_indices,     # num_starts
    M.phi,               # num_starts x N
    val_start_ids=splits["val_starts"],
    how_many=3,
    geod_rows=geod_rows,     
    title_prefix="Mobius Strip | val",
)