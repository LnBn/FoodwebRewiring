"""
kondoh_parallel_save.py
Simulation of Kondoh's model (2003, Science) with random and cascade food webs.
Parallelization with joblib and automatic saving of results (figures and CSV).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

# ==============================
#  KONDOH MODEL
# ==============================

def generate_foodweb(N, C, model='random', levels=3):
    """
    Generates an adjacency matrix A_ij where A[i,j] = 1 if species i consumes species j.
    The resulting connectance approximates the desired C.
    model: 'random' or 'cascade'
    """

    A = np.zeros((N, N))
    possible_links = []

    if model == 'random':
        # All i != j pairs are allowed
        possible_links = [(i, j) for i in range(N) for j in range(N) if i != j]

    elif model == 'cascade':
        trophic_levels = np.random.randint(0, levels, N)
        # Only species at higher trophic levels can feed on lower ones
        possible_links = [
            (i, j) for i in range(N) for j in range(N)
            if trophic_levels[i] >= trophic_levels[j]
        ]

    else:
        raise ValueError("model must be 'random' or 'cascade'")

    # Maximum number of possible links
    max_links = len(possible_links)
    L = int(round(C * N * (N - 1)))
    L = min(L, max_links)

    if L > 0:
        chosen_links = np.random.choice(max_links, L, replace=False)
        for idx in chosen_links:
            i, j = possible_links[idx]
            A[i, j] = 1

    np.fill_diagonal(A, 0)
    return A


def assign_parameters(A):
    """
    Assigns biologically consistent parameters:
    - r_i = 0 for consumers, > 0 for producers (??)
    - f_ij random where A[i,j] = 1
    - s_i = 1 for all species
    """
    N = A.shape[0]
    s = np.ones(N)
    f = np.random.uniform(0.1, 1.0, (N, N)) * A   # efficiencies only on existing links
    e = 0.15

    # producers: species with no resources (no j such that A[i,j] = 1)
    producers = np.where(A.sum(axis=1) == 0)[0]

    r = np.random.uniform(0.00001, 0.1, N)  # every species gets a growth rate

    return r, s, f, e


def get_resource_consumer(A):
    """ Given adjacency matrix A, return the list of resources and consumers. """
    n = A.shape[0]
    resource = [list(np.where(A[i, :] == 1)[0]) for i in range(n)]
    consumer = [list(np.where(A[:, i] == 1)[0]) for i in range(n)]
    return resource, consumer


def select_adaptive_species(A, F=0.8):
    """ Selects a fraction F of consumers as adaptive foragers. """
    resource, consumer = get_resource_consumer(A)
    consumer_idx = [i for i, r in enumerate(resource) if len(r) > 0]
    n_consumers = len(consumer_idx)

    if n_consumers == 0 or F <= 0:
        return np.zeros(len(resource))

    n_adaptive = min(int(round(F * n_consumers)), n_consumers)
    selected = np.random.choice(consumer_idx, n_adaptive, replace=False)
    G = np.zeros(len(resource))
    G[selected] = 0.25
    return G



def kondoh_model(y, t, N, r, s, e, f, G, F, A):
    X = y[:N].copy()
    X = np.where(X < 1e-13, 0.0, X)
    a = y[N:].reshape((N, N)).copy()  # foraging effort matrix

    dX = np.zeros(N)
    da = np.zeros((N, N))
    resource, consumer = get_resource_consumer(A)

    for i in range(N):
        gain = sum(e * f[i, j] * a[i, j] * X[j] for j in resource[i])
        loss = sum(f[j, i] * a[j, i] * X[j] for j in consumer[i])
        dX[i] = X[i] * (r[i] - s[i] * X[i] + gain - loss)

        # Adaptive update only if F > 0
        if F > 0 and G[i] > 0:
            avg_profit = sum(a[i, k] * e * f[i, k] * X[k] for k in resource[i])
            for j in resource[i]:
                da[i, j] = G[i] * a[i, j] * (e * f[i, j] * X[j] - avg_profit)

    return np.concatenate([dX, da.flatten()])


def community_persistence(N, C, F, model='cascade', t_max=1e5):
    """
    Simulates one food web and returns True if all species persist (Xi > 1e-13)
    """
    # --- Build network ---
    A = generate_foodweb(N, C, model)
    r, s, f, e = assign_parameters(A)
    G = select_adaptive_species(A, F)

    # --- Initial conditions ---
    X0 = np.random.uniform(0.01, 0.1, N)
    a0 = A.copy()
    a0 = a0 / np.maximum(a0.sum(axis=1, keepdims=True), 1)

    y0 = np.concatenate([X0, a0.flatten()])
    t = np.linspace(0, t_max, 2000)

    try:
        sol = odeint(kondoh_model, y0, t,
                     args=(N, r, s, e, f, G, F, A), mxstep=10000)
        if np.any(np.isnan(sol)):
            return 0
    except Exception:
        return 0

    X_final = sol[-1, :N]
    return np.all(X_final > 1e-13)


# ==============================
#  PARALLELIZED EXPERIMENT
# ==============================

def run_heatmap(model_type='random', num_jobs=-1, save=True):
    N_vals = np.arange(2, 22, 5)
    C_vals = np.linspace(0.1, 1, 5)
    num_simulations = 10
    F = 1.0
    t_max = 100000
    heatmap = np.zeros((len(N_vals), len(C_vals)))

    for i, N in enumerate(N_vals):
        for j, C in enumerate(C_vals):
            results = Parallel(n_jobs=num_jobs)(
                delayed(community_persistence)(
                    N, C, F, model=model_type, t_max=t_max
                )
                for _ in tqdm(range(num_simulations),
                              desc=f"{model_type}: N={N}, C={C:.2f}")
            )
            heatmap[i, j] = np.mean(results)

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, origin='lower',
               extent=[C_vals[0], C_vals[-1], N_vals[0], N_vals[-1]],
               aspect='auto', cmap='jet', vmin=0, vmax=1)
    plt.colorbar(label='Fraction of surviving species')
    plt.xlabel('Connectance (C)')
    plt.ylabel('Number of species (N)')
    plt.title(f'Mean Persistence ({model_type} food web)')

    # --- Save ---
    if save:
        os.makedirs("figures", exist_ok=True)
        fig_path = f"figures/heatmap_{model_type}_F={F}.png" #change to save figures
        csv_path = f"figures/heatmap_{model_type}_F={F}.csv"
        plt.savefig(fig_path, dpi=300)
        pd.DataFrame(heatmap, index=N_vals,
                     columns=np.round(C_vals, 2)).to_csv(csv_path)
        print(f"✅ Figure saved in: {fig_path}")
        print(f"✅ Data saved in:   {csv_path}")

    plt.show()
    plt.close()


# ==============================
#  MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    for model_type in ['cascade', 'random']:
        run_heatmap(model_type=model_type, num_jobs=-1, save=True)
