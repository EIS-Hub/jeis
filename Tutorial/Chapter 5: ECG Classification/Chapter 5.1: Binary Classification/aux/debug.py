import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from aux.training_utils import j_v_prediction_per_sample


def out_debug(batch_spikes, batch_lbls, hyperparams, w):
    (S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o) = j_v_prediction_per_sample(w, batch_spikes, hyperparams)
    # V_mem shape (batch_size, 143, 2) in elija's case
    out = V_o.max(axis=1)
    # computing loss
    logit = jax.nn.softmax(out, axis=1)
    loss = -jnp.mean((jnp.log(logit[jnp.arange(batch_spikes.shape[0]), batch_lbls])))
    pred = jnp.argmax(out, axis=1)
    acc = jnp.count_nonzero(pred == batch_lbls)/len(batch_lbls)
    print(f'loss: {loss}, acc: {acc}')
    return ((S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o)), out, loss, pred


def plot_net(w, w_original, batch_spikes, batch_labels, hyperparams, sample_id):
    print(f'w_original[0] == w[0]: {np.array_equal(w_original[0], w[0])}')

    *state_orig, out_orig, loss_orig, pred_orig = (
        out_debug(batch_spikes, batch_labels, hyperparams, w_original)
    )
    *state, out, loss, pred = (
        out_debug(batch_spikes, batch_labels, hyperparams, w)
    )

    (S_h1_orig, I_h1_orig, V_h1_orig), (S_h2_orig, I_h2_orig, V_h2_orig), (S_o_orig, I_o_orig, V_o_orig) = state_orig[0]
    (S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o) = state[0]
    # create a 3 x 1 figure
    # on the top    plot the first 10 samples of input spikes
    batch_spikes_to_plot = np.zeros(batch_spikes.shape)
    batch_spikes_to_plot[:, :, 0] = jnp.where(batch_spikes[:, :, 0] == 1, 1, np.nan)
    batch_spikes_to_plot[:, :, 1] = jnp.where(batch_spikes[:, :, 1] == 1, 1, np.nan)
    batch_spikes_to_plot[:, :, 2] = jnp.where(batch_spikes[:, :, 2] == 1, 1, np.nan)
    batch_spikes_to_plot[:, :, 3] = jnp.where(batch_spikes[:, :, 3] == 1, 1, np.nan)
    S_h1_to_plot = jnp.where(S_h1 == 1, 1, np.nan)
    S_h1_orig_to_plot = jnp.where(S_h1_orig == 1, 1, np.nan)
    S_h2_to_plot = jnp.where(S_h2 == 1, 1, np.nan)
    S_h2_orig_to_plot = jnp.where(S_h2_orig == 1, 1, np.nan)
    S_o_to_plot = jnp.where(S_o == 1, 1, np.nan)
    S_o_orig_to_plot = jnp.where(S_o_orig == 1, 1, np.nan)

    fig, axs = plt.subplots(7, 2, figsize=(14, 6), sharex=True, sharey='row')
    for i in range(4):
        axs[0, 0].plot(batch_spikes_to_plot[sample_id, :, i] * i, 'o', markersize=2, color='b')
        axs[0, 1].plot(batch_spikes_to_plot[sample_id, :, i] * i, 'o', markersize=2, color='b')
    for i in range(64):
        axs[1, 0].plot(I_h1[sample_id, :, i], label='I_h')
        axs[1, 1].plot(I_h1_orig[sample_id, :, i], label='I_h_orig')
        axs[2, 0].plot(V_h1[sample_id, :, i], label=f'V_h {i}')
        axs[2, 1].plot(V_h1_orig[sample_id, :, i], label=f'V_h_orig {i}')
        axs[3, 0].plot(S_h1_to_plot[sample_id, :, i] * i, 'o', markersize=2, label=f'neuron {i}')
        axs[3, 1].plot(S_h1_orig_to_plot[sample_id, :, i] * i, 'o', markersize=2, label=f'neuron {i}')
        axs[4, 0].plot(S_h2_to_plot[sample_id, :, i] * i, 'o', markersize=2, label=f'neuron {i}')
        axs[4, 1].plot(S_h2_orig_to_plot[sample_id, :, i] * i, 'o', markersize=2, label=f'neuron {i}')
        axs[5, 0].plot(S_o_to_plot[sample_id, :, i] * i, 'o', markersize=2, label=f'neuron {i}')
        axs[5, 1].plot(S_o_orig_to_plot[sample_id, :, i] * i, 'o', markersize=2, label=f'neuron {i}')
    for i in range(4):
        axs[6, 0].plot(V_o[sample_id, :, i], label=f'V_o {i}')
        axs[6, 1].plot(V_o_orig[sample_id, :, i], label=f'V_o_orig {i}')
    axs[0, 0].set_title('Trained')
    axs[0, 1].set_title('Original')
    axs[0, 0].set_ylabel('Input spikes')
    axs[1, 0].set_ylabel('I_h1')
    axs[2, 0].set_ylabel('V_h1')
    axs[3, 0].set_ylabel('S_h1')
    axs[4, 0].set_ylabel('S_h2')
    axs[5, 0].set_ylabel('S_o')
    axs[6, 0].set_ylabel('V_o')
    axs[6, 0].set_xlabel('Time step')
    axs[6, 1].set_xlabel('Time step')
    for i in range(7):
        axs[i, 1].tick_params(left=True, labelleft=True)
    plt.show()