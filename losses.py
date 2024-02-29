import jax.numpy as np
from jax.nn import log_softmax


def fr_decoder(spikes):
    """ Calculates log-softmax based on spike count logits. """
    Yhat = log_softmax(np.sum(spikes, 1))
    return Yhat