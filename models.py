import jax
import jax.numpy as jnp

@jax.custom_jvp
def gr_than(x, thr):
    """ Thresholding function for spiking neurons. """
    return (x > thr).astype(jnp.float32)


@gr_than.defjvp
def gr_jvp(primals, tangents):
    """ Surrogate gradient function for thresholding. """
    x, thr = primals
    x_dot, y_dot = tangents
    primal_out = gr_than(x, thr)
    tangent_out = x_dot / (10 * jnp.absolute(x - thr) + 1)**2
    return primal_out, tangent_out

def lif_forward(state, input_spikes):
    ''' Vectorized Leaky Integrate and Fire (LIF) neuron model
    '''
    w, (i, v, z) = state[0]
    tau_mem, v_th, timestep = state[1]
    i = jnp.dot(w, input_spikes)  # + jnp.dot(Wrec, S_h)
    v = (1 - timestep / tau_mem) * v + i - z * v_th
    v = jnp.maximum(0, v)
    z = gr_than(v, v_th)

    return ((w, (i, v, z)), state[1]), (i, v, z)


def rlif_forward(state, x):
    ''' Vectorized Recurrent Leaky Integrate and Fire (LIF) neuron model
    '''
    inp_weight, rec_weight, bias, out_weight = state[0]     # Static weights
    thr_rec, thr_out, alpha, kappa = state[1]         # Static neuron states
    v, z, vo, zo = state[2]                           # Dynamic neuron states

    v  = alpha * v + jnp.matmul(x, inp_weight) + jnp.matmul(z, rec_weight) + bias - z * thr_rec
    z = gr_than(v, thr_rec)

    vo = kappa * vo + jnp.matmul(z, out_weight)
    # zo = gr_than(vo, thr_out)

    return [[inp_weight, rec_weight, bias, out_weight], [thr_rec, thr_out, alpha, kappa], [v, z, vo, zo]], [z, vo]