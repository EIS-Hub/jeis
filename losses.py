import jax.numpy as jnp
import jax.nn 
from jax.nn import log_softmax

def decoder_sum( out_v_mem ):
    ''' Decodes the output as the sum of the membrane voltage over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( jnp.mean( out_v_mem, axis=1 ), axis=-1 )

def decoder_cum( out_v_mem ):
    ''' Decodes the output as the sum of the "softmaxed" membrane voltage over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jnp.mean( jax.nn.softmax( out_v_mem, axis=-1 ), axis=1)

def decoder_vmax( out_v_mem ):
    ''' Decodes the output as the maximum of the membrane voltage over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( jnp.max( out_v_mem, axis=1 ), axis=-1 )

def decoder_vlast( out_v_mem ):
    ''' Decodes the output as the last value of the membrane voltage over time '''
    # out_v_mem dims: [batch, time_steps, out_dim]
    return jax.nn.softmax( out_v_mem[:,-1], axis=-1 )

def fr_decoder(spikes):
    """ Calculates log-softmax based on spike count logits. """
    Yhat = log_softmax(jnp.sum(spikes, 1))
    return Yhat

@jit
def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()