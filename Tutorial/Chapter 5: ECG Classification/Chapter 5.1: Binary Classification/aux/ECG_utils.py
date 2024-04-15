import torch
import numpy as np
import jax


class DummyArgs:
    def __init__(self):
        self.batch_size = 32
        self.lr = 0.005
        self.num_epochs = 1
        self.tau_mem = 0.15
        self.tau_mem_o = 0.10
        self.Vth = 1
        self.timestep = 1/360
        self.num_hidden = 64
        self.weight_gain = 0.4
        self.seed = 0


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))

    labels = np.array(transposed_data[1])
    spikes = np.array(transposed_data[0])

    return spikes, labels


# initialize the weights
def init_weights_2layer(key, n_in, n_rec1, n_out, gain):
    n_rec2 = int(n_rec1*0.5)
    key, subkey_in, subkey_rec, subkey_out = jax.random.split(key, 4)
    W1 = jax.random.normal(subkey_in, (n_rec1, n_in)) * (gain/np.sqrt(n_in))
    Wrec1 = jax.random.normal(subkey_rec, (n_rec1, n_rec1)) * (gain/np.sqrt(n_rec1))
    W2 = jax.random.normal(subkey_rec, (n_rec2, n_rec1)) * (gain/np.sqrt(n_rec1))
    Wrec2 = jax.random.normal(subkey_rec, (n_rec2, n_rec2)) * (gain/np.sqrt(n_rec2))
    W3 = jax.random.normal(subkey_out, (n_out, n_rec2)) * (gain/np.sqrt(n_rec2))
    return key, (W1, Wrec1, W2, Wrec2, W3)