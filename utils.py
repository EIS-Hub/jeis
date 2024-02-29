import jax
import jax.numpy as jnp
from models import rlif_forward    

def net_step(net_params, x_t):
        ''' Single time step network inference (x_t => yhat_t)
        '''
        net_params, [z_rec, v_out] = rlif_forward(net_params, x_t)
        return net_params, [z_rec, v_out]

#LSUV initialization
'''
Implementation of Layer-sequential unit-variance (LSUV) initialization
https://doi.org/10.48550/arXiv.1511.06422
'''
def data_driven_init(state, x):
    tgt_mean =-0.52
    tgt_var =1.
    mean_tol =.1
    var_tol =.1
    done = False
    
    net_state = net_step(state, x)[0]

    while not done:
        done = True
        
        net_state = net_step(net_state, x)[0] #Data pass without weight updates
        
        inp_weight, rec_weight, bias, out_weight = net_state[0]
        weights = [inp_weight, rec_weight, out_weight]
        # bias_arr = [bias]

        neuron_dyn = net_state[2]
        V = [neuron_dyn[0], neuron_dyn[2]]

        v=0    
        for i in range(0,len(weights)):
            if i == len(weights)+1:
                v+=1
            done=True

            st = V[v] #Membrane potential of the current layer
            kernel = jnp.array(weights[i])

            st = jnp.array(st)
            
            ## Adjusting weights to unit variance
            if jnp.abs(jnp.var(st) - tgt_var) > var_tol:
                kernel = jnp.multiply(jnp.sqrt(tgt_var)/jnp.sqrt(jnp.maximum(jnp.var(st),1e-2)), kernel)
                weights[i] = kernel
                done *= False
            else:
                done *=True

            if v == 1:
                continue

            ## Adjusting bias to target mean
            bias = jnp.array(bias)

            if jnp.abs(jnp.mean(st) - tgt_mean) > mean_tol:
                bias = jnp.subtract(bias, .15*(jnp.mean(st) - tgt_mean))
                done *= False
            else:
                done *= True

        inp_weight, rec_weight, out_weight = weights
        net_state = [[inp_weight, rec_weight, bias, out_weight], state[1], state[2]]

    return net_state

# dropout function
def dropout(rng, x, rate):
    keep = jax.random.bernoulli(rng, 1 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0)

# Noise Injection
@jax.custom_jvp
def add_noise(w, key, noise_std):
    ''' Adds noise only for inference 
    w: weight [array]
    key: Pseudo Random Number Generator
    noise_std: standard deviation of the noise to be injected'''
    noisy_w = jnp.where(w != 0.0,
                        w + jax.random.normal(key, w.shape) * jnp.max(jnp.abs(w)) * noise_std,
                        w)
    return noisy_w

@add_noise.defjvp
# custom backward
def add_noise_jvp(primals, tangents):
    weight, key, noise_std = primals
    x_dot, y_dot, z_dot = tangents
    primal_out = add_noise(weight, key, noise_std)
    tangent_out = x_dot
    return primal_out, tangent_out