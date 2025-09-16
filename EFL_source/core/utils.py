import pickle
from abc import abstractmethod

import numpy as np
import torch
from matplotlib import pyplot as plt

# Calculate model accuracy

random_gn = np.random.default_rng(46)

def channel_gain(transmitter, receiver):
    beta_0 = -30  # -30db
    d_0 = 1  # 1m
    alpha = 3
    # initialize with rayleigh fading based on complex normal random
    rayleigh_faded_channel_matrix = random_gn.standard_normal((1, 2))
    # Calculating beta(t)

    transmitter_receiver_distance_real = np.math.sqrt(
        np.power(receiver.position_m_real[0] - transmitter.position_m[0], 2) + np.power(
            receiver.position_m_real[1] - transmitter.position_m[1], 2))

    transmitter_receiver_distance_pred = np.math.sqrt(
        np.power(receiver.position_m[0] - transmitter.position_m[0], 2) + np.power(
            receiver.position_m[1] - transmitter.position_m[1], 2))
    # transmit power in db
    clear_transmit_power_real = beta_0 - 10 * alpha * np.log10(transmitter_receiver_distance_real / d_0)
    clear_transmit_power_pred = beta_0 - 10 * alpha * np.log10(transmitter_receiver_distance_pred / d_0)

    # convert to watt
    clear_transmit_power_real = np.sqrt(10 ** (clear_transmit_power_real / 10))
    clear_transmit_power_pred = np.sqrt(10 ** (clear_transmit_power_pred / 10))

    # applying rayleigh fading
    rayleigh_faded_channel_matrix_real = rayleigh_faded_channel_matrix * clear_transmit_power_real
    rayleigh_faded_channel_matrix_pred = rayleigh_faded_channel_matrix * clear_transmit_power_pred

    return norm2_power(rayleigh_faded_channel_matrix_real), norm2_power(rayleigh_faded_channel_matrix_pred)


def norm2_power(rayleigh_faded_channel_matrix):
    return rayleigh_faded_channel_matrix[:, 0] ** 2 + rayleigh_faded_channel_matrix[:, 1] ** 2


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)



# Dirichlet non-IID split that ensures no empty client
def dirichlet_split_strict(labels, num_clients, alpha, seed=45):
    rng = np.random.default_rng(seed)
    y = np.array(labels)
    n_classes = int(y.max()) + 1
    idx_by_class = [np.where(y == k)[0] for k in range(n_classes)]
    # resample until every client gets at least one index
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for idx_k in idx_by_class:
            rng.shuffle(idx_k)
            p = rng.dirichlet(alpha * np.ones(num_clients))
            p = p / p.sum()
            counts = (p * len(idx_k)).astype(int)
            # distribute remainder
            for i in range(len(idx_k) - counts.sum()):
                counts[i % num_clients] += 1
            start = 0
            for i, c in enumerate(counts):
                if c > 0:
                    client_indices[i].extend(idx_k[start:start+c].tolist())
                    start += c
        sizes = np.array([len(ci) for ci in client_indices])
        if (sizes > 0).all():
            return [np.array(ci, dtype=int) for ci in client_indices]


def plot_dirichlet_allocation(client_indices, labels, num_classes, title=None):
    """
    Visualize the class mixture per client produced by a Dirichlet split.
    - client_indices: list of arrays (indices per client)
    - labels: full dataset labels (e.g., trainset.targets)
    - num_classes: e.g., 10 for CIFAR-10
    """
    num_clients = len(client_indices)
    y = np.asarray(labels)

    # counts[c, k] = #samples of class k on client c
    counts = np.zeros((num_clients, num_classes), dtype=int)
    for c, idxs in enumerate(client_indices):
        cls, cnt = np.unique(y[idxs], return_counts=True)
        counts[c, cls] = cnt

    sizes = counts.sum(axis=1)
    props = counts / np.clip(sizes[:, None], 1, None)  # proportions per client

    # --- stacked bar: per-client class proportions ---
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(num_clients)
    bottom = np.zeros(num_clients)
    cmap = plt.cm.get_cmap("tab10", num_classes)

    for k in range(num_classes):
        ax.bar(x, props[:, k], bottom=bottom, color=cmap(k), edgecolor="none", label=f"class {k}")
        bottom += props[:, k]

    ax.set_title(title or "Dirichlet class proportions per client")
    ax.set_xlabel("Client")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    # ax.legend(ncol=min(num_classes, 10), fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.legend(
        ncol=1,  # <- vertical (one column)
        fontsize=8,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Class",
        frameon=False
    )
    fig.tight_layout()

    # --- bar: dataset size per client ---
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(x, sizes, width=0.9)
    ax2.set_title("Number of samples per client")
    ax2.set_xlabel("Client")
    ax2.set_ylabel("# samples")
    fig2.tight_layout()

    return counts, props, sizes
