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
    tangent_out = x_dot / (10 * jnp.absolute(x - thr) + 1) ** 2
    return primal_out, tangent_out

# 2-layer version
def lif_rec_forward_2layer(state, input_spikes):
    (W1, Wrec1, W2, Wrec2, W3), (S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o) = state[0]
    # TODO : W1, Wrec1, W2, Wrec2, W3
    tau_mem, tau_mem_o, Vth, timestep = state[1]
    I_h1 = jnp.dot(W1, input_spikes)  # + jnp.dot(Wrec, S_h)
    V_h1 = (1 - timestep / tau_mem) * V_h1 + I_h1 - S_h1 * Vth
    V_h1 = jnp.maximum(0, V_h1)
    S_h1 = gr_than(V_h1, Vth)
    
    I_h2 = jnp.dot(W2, S_h1)
    V_h2 = (1 - timestep / tau_mem) * V_h2 + I_h2 - S_h2 * Vth
    V_h2 = jnp.maximum(0, V_h2)
    S_h2 = gr_than(V_h2, Vth)
    
    I_o = jnp.dot(W3, S_h2)
    V_o = (1 - timestep / tau_mem_o) * V_o + I_o
    # V_o = jnp.maximum(0, V_o)
    S_o = gr_than(V_o, Vth)

    return (((W1, Wrec1, W2, Wrec2, W3), (S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o)), state[1]), ((S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o))
