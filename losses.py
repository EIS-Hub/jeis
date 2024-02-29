import jax.numpy as jnp
from jax.nn import log_softmax


def fr_decoder(spikes):
    """ Calculates log-softmax based on spike count logits. """
    Yhat = log_softmax(jnp.sum(spikes, 1))
    return Yhat

@jit
def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()