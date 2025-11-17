import math
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
# Optional progress bar; becomes a no-op if tqdm isn't installed
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs): return x

def neighbors(r, c, n):
    return [((r-1) % n, c % n), ((r+1) % n, c % n), (r % n, (c-1) % n), (r % n, (c+1) % n)]

def _simulate_walks_chunk(n, rr, cc, chunk_walks, p_term, gamma, sigma, seed):
    """
    Worker: simulate `chunk_walks` random walks starting at (rr, cc),
    and return a (n, n) array with accumulated contributions.
    """
    rng = np.random.default_rng(seed)
    v_local = np.zeros((n, n), dtype=np.float64)
    s = (sigma * sigma) / 2. # same as your code

    for _ in range(chunk_walks):
        r, c = rr, cc
        load = 1.0
        fk = 1.0
        wsurv = 1.0
        k = 0

        while True:
            # contribute k-th term
            v_local[r, c] += (load * fk * wsurv)

            # geometric termination before attempting next step
            if rng.random() < p_term:
                break

            # survived: update survival debias
            wsurv *= 1.0 / (1.0 - p_term)

            # one RW step (unbiased expectation for Markov operator)
            nbrs = neighbors(r, c, n)
            dv = len(nbrs)
            wr, wc = nbrs[rng.integers(dv)]
            dw = len(neighbors(wr, wc, n))  # general graphs; 4 on the torus grid
            p = 1.0 / dv
            u = gamma / math.sqrt(dv * dw)
            load *= (u / p)

            r, c = wr, wc

            # update factorial weight to s^{k+1}/(k+1)!
            k += 1
            fk *= s / k

    return v_local


def generate_signature_vector_diffusion_sym(
    n=10, source=(0, 0), num_walks=400,
    p_term=0.2, gamma=1.0, sigma=1.0, seed=0,
    workers=None, chunk_size=None, show_progress=True
):
    """
    Parallel version (drop-in compatible). Set `workers>1` to enable multiprocessing.

    Args:
        n, source, num_walks, p_term, gamma, sigma, seed: same semantics as your original.
        workers: number of processes (default: os.cpu_count()).
        chunk_size: walks per task; default balances overhead vs. latency.
        show_progress: show a tqdm progress bar over chunks.

    Returns:
        Flattened (n*n,) signature vector.
    """
    rr, cc = source
    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    # Sequential fast path (identical to original behavior)
    if workers <= 1 or num_walks <= 1:
        v_local = _simulate_walks_chunk(n, rr, cc, num_walks, p_term, gamma, sigma, seed)
        return (v_local / num_walks).reshape(-1)

    # Create chunk sizes
    if chunk_size is None:
        # heuristic: ~4 chunks per worker
        chunk_size = max(1, num_walks // (workers * 4) or 1)

    chunk_counts = []
    remaining = num_walks
    while remaining > 0:
        take = min(chunk_size, remaining)
        chunk_counts.append(take)
        remaining -= take

    # Derive a reproducible seed for each chunk
    ss = np.random.SeedSequence(seed)
    child_ss = ss.spawn(len(chunk_counts))
    child_seeds = [int(cs.generate_state(1, dtype=np.uint64)[0]) for cs in child_ss]

    v_total = np.zeros((n, n), dtype=np.float64)

    # Submit tasks
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _simulate_walks_chunk, n, rr, cc, walks, p_term, gamma, sigma, sd
            )
            for walks, sd in zip(chunk_counts, child_seeds)
        ]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures),
                            desc=f"Generating signature vector N={n} (parallel)",
                            leave=False)

        for fut in iterator:
            v_total += fut.result()

    return (v_total / num_walks).reshape(-1)

def _wrap_deltas(n, i0):
    """Minimal torus distance from index vector 0..n-1 to i0."""
    idx = np.arange(n)
    d = np.abs(idx - i0)
    return np.minimum(d, n - d)

def signature_vector_choro_center(n, sigma=1., plot=False):
    """
    Center signature vector using your 'choro' normalization:
    sqrt(2/pi) * exp(- (dr^2 + dc^2)).
    Returns an (n, n) array.
    """
    r0 = c0 = n // 2
    dr = _wrap_deltas(n, r0)                 # shape (n,)
    dc = _wrap_deltas(n, c0)                 # shape (n,)
    # separability: exp(-D2) = exp(-dr^2) * exp(-dc^2)
    sig = (2/np.pi)**0.5 * np.outer(np.exp(-dr**2/(sigma**2)), np.exp(-dc**2/(sigma**2)))
    if plot:
        plt.imshow(sig, origin="lower")
    return sig

def signature_vector_dwip_center(n, sigma=1., plot=False):
    """
    Center signature vector using your 'dwip' normalization:
    (1 / (2*pi)) * exp(- (dr^2 + dc^2) / 2).
    Returns an (n, n) array.
    """
    r0 = c0 = n // 2
    dr = _wrap_deltas(n, r0)
    dc = _wrap_deltas(n, c0)
    sig = (1/(2*np.pi*sigma**2)) * np.outer(np.exp(-dr**2/(2*sigma**2)), np.exp(-dc**2/(2*sigma**2)))
    if plot:
        plt.imshow(sig, origin="lower")
    return sig

# K^{1/2} row (time halved): variance parameter is σ^2 (not 2σ^2)
def signature_vector_ananya_center(n, sigma):
    r0 = c0 = n//2
    dr = _wrap_deltas(n, r0)
    dc = _wrap_deltas(n, c0)
    D2 = dr[:,None]**2 + dc[None,:]**2
    # 2D heat kernel at time t/2 = σ^2/4: 1/(4π(t/2)) * exp(-D2 / (4(t/2)))
    # => 1/(π σ^2) * exp(-D2 / (σ^2))
    return (1./(2*np.pi*sigma*sigma)) * np.exp(-D2/(2*sigma*sigma))

def mse_signature_vectors(n, sv1, sv2):
    sv1_vec = sv1.reshape(-1)
    sv2_vec = sv2.reshape(-1)
    return np.mean((sv1_vec - sv2_vec) ** 2)

if __name__ == "__main__":

    mses_choro = []
    mses_dwip = []
    mses_ananya = []
    Ns = [16, 32, 64]
    reps_list = [400000, 800000, 1600000]
    for n, reps in zip(Ns, reps_list):
        sigma = 2.
        signature_vector = generate_signature_vector_diffusion_sym(n=n, sigma=sigma, source=(n // 2, n // 2), num_walks=reps, p_term=0.01, gamma=1.0, seed=346511053).reshape(n,n)
        mses_choro.append(mse_signature_vectors(n, signature_vector, signature_vector_choro_center(n,sigma=sigma)))
        mses_dwip.append(mse_signature_vectors(n, signature_vector, signature_vector_dwip_center(n,sigma=sigma)))
        mses_ananya.append(mse_signature_vectors(n, signature_vector, signature_vector_ananya_center(n,sigma=sigma*np.sqrt(1/3.4))))
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(signature_vector, origin='lower')
        plt.title(f'Signature Vector n={n}, reps={reps}')
        plt.colorbar()
        plt.subplot(1, 4, 2)
        plt.imshow(signature_vector_choro_center(n,sigma=sigma), origin='lower')
        plt.title(f'Choromanski Signature Vector n={n}')
        plt.colorbar()
        plt.subplot(1, 4, 3)
        plt.imshow(signature_vector_dwip_center(n,sigma=sigma), origin='lower')
        plt.title(f'DWIP Signature Vector n={n}')
        plt.colorbar()
        plt.subplot(1, 4, 4)
        plt.imshow(signature_vector_ananya_center(n,sigma=sigma*np.sqrt(1/3.4)), origin='lower')
        plt.title(f'ANANYA Signature Vector n={n}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'signature_vector_n{n}_reps{reps}.png')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(Ns, mses_choro, marker='o', label='Choromanski et al. signature vector MSE', color='blue')
    plt.plot(Ns, mses_dwip, marker='s', label='DWIP signature vector MSE', color='green')
    plt.plot(Ns, mses_ananya, marker='^', label='ANANYA signature vector MSE', color='red')
    plt.xlabel('n (grid size n x n)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Signature Vector MSE Comparison')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(Ns, mses_choro, marker='o', label='Choromanski et al. signature vector MSE', color='blue')
    plt.plot(Ns, mses_ananya, marker='^', label='ANANYA signature vector MSE', color='red')
    plt.xlabel('n (grid size n x n)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Signature Vector MSE Comparison')
    plt.legend()
    plt.savefig('signature_vector_mse_comparison.png')
    mses_df = pd.DataFrame({
        'n': Ns,
        'MSE_Choro': mses_choro,
        'MSE_DWIP': mses_dwip,
        'MSE_Ananya': mses_ananya
    })
    print(mses_df)