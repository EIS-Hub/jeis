import jax
import jax.numpy as jnp
from aux.model import lif_rec_forward_2layer


def prediction_per_sample(w, single_input, hyperparams, mulLayer=True):
    # single_input shape (n_steps, n_in)
    V_h1 = jnp.zeros((w[0].shape[0],))
    I_h1 = jnp.zeros((w[0].shape[0],))
    S_h1 = jnp.zeros((w[0].shape[0],))
    V_h2 = jnp.zeros((w[2].shape[0],))
    I_h2 = jnp.zeros((w[2].shape[0],))
    S_h2 = jnp.zeros((w[2].shape[0],))
    V_o = jnp.zeros((w[4].shape[0],))
    I_o = jnp.zeros((w[4].shape[0],))
    S_o = jnp.zeros((w[4].shape[0],))
    state = ((w, (S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o)), hyperparams)
    
    # define here if you want to use lif_forward or lif_rec_forward!
    _, ((S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o)) = jax.lax.scan(lif_rec_forward_2layer, state, single_input) # multi layer
    #_, ((S_h, I_h, V_h), (S_o, I_o, V_o)) = jax.lax.scan(lif_rec_forward, state, single_input) # multi layer
    #_, ((S_h, I_h, V_h), (S_o, I_o, V_o)) = jax.lax.scan(lif_forward, state, single_input) # single layer

    #return (S_h, I_h, V_h),  (S_o, I_o, V_o)
    return (S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o)
    
v_prediction_per_sample = jax.vmap(prediction_per_sample, in_axes=(None, 0, None))
j_v_prediction_per_sample = jax.jit(v_prediction_per_sample, static_argnums=(2,))


def f1_fn(y_pred, y_true):
    # for every class, compute class-specific f1 score
    # multiclass: find out how many different classes exist in y_true (y_true is a jnparray)#
    #nb_classes = len(jnp.unique(y_true))
    # compute f1 score for each class
    f1_final = jnp.zeros((5,))
    precision_final = jnp.zeros((5,))
    recall_final = jnp.zeros((5,))
    for i in range(5): # 5 possible classes
        # get true positives (tp), false positives (fp), false negatives (fn)
        tp = jnp.sum((y_pred == i) & (y_true == i))
        fp = jnp.sum((y_pred == i) & (y_true != i))
        fn = jnp.sum((y_pred != i) & (y_true == i))
        #print(f'tp: {tp}, fp: {fp}, fn: {fn}, class: {i}')
        # compute precision, recall, f1
        precision = jnp.where((tp + fp) > 0,tp / (tp + fp),0.0) # if no positive predictions, precision is 0
        recall = jnp.where((tp + fn) > 0,tp / (tp + fn),1.0) # if no positive samples, recall is 1
        f1 = jnp.where((precision + recall) > 0,2 * (precision * recall) / (precision + recall),0.0) # if precision and recall are 0, f1 is 0
        # if no positive samples, f1, recall, precision should be nan
        f1 = jnp.where((tp + fn) == 0,jnp.nan,f1)
        recall = jnp.where((tp + fn) == 0,jnp.nan,recall)
        precision = jnp.where((tp + fp) == 0,jnp.nan,precision)
        #print(f'f1: {f1}, precision: {precision}, recall: {recall}, class: {i}')

        f1_final = f1_final.at[i].set(f1)
        precision_final = precision_final.at[i].set(precision)
        recall_final = recall_final.at[i].set(recall)
        # exclude nan values from the mean using jax
    return jnp.nanmean(f1_final), jnp.nanmean(precision_final), jnp.nanmean(recall_final)

# jit the f1 function
f1_fn_jit = jax.jit(f1_fn)

def loss_fn(w, batch_spikes, hyperparams, batch_lbls): # there was a comma after batch_lbls
    # batch_spikes shape (batch_size, n_steps, n_in)
    (S_h1, I_h1, V_h1), (S_h2, I_h2, V_h2), (S_o, I_o, V_o) = j_v_prediction_per_sample(w, batch_spikes, hyperparams)
    # V_mem shape (batch_size, 143, 2) in elija's case
    out = V_o.max(axis=1)
    # computing loss
    logit = jax.nn.softmax(out, axis=1)
    # fr_loss = jnp.mean(10 * (jnp.sum(S_h, axis=1).mean(axis=1) - 8)) ** 2
    loss = -jnp.mean((jnp.log(logit[jnp.arange(batch_spikes.shape[0]), batch_lbls]))) # + 0.01 * fr_loss
    # computing acc
    pred = jnp.argmax(out, axis=1)
    acc = jnp.count_nonzero(pred == batch_lbls) / len(batch_lbls)

    # implement f1 loss for multiclass classification
    # see: https://www.baeldung.com/cs/multi-class-f1-score
    f1, precision, recall = f1_fn_jit(pred, batch_lbls)
    f1_loss = 1-f1
    loss = loss + f1_loss

    return loss, (acc, f1, recall, precision)


loss_fn_jit = jax.jit(loss_fn, static_argnums=(2,))


def update_opt(w, input_spikes, hyperparams, gt_y):
    (loss, (acc, f1, rec, prec)), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        w, input_spikes, hyperparams, gt_y,
    )
    return (loss, (acc, f1, rec, prec)), grad

update_opt_jit = jax.jit(update_opt, static_argnums=(2,))



def run_epoch(key, loader, hyperparams, opt_state, get_params, e=None, opt_update=None):
    local_loss = []
    local_acc = []
    local_prec = []
    local_rec = []
    local_f1 = []
    for idx, batch in enumerate(loader):
        w = get_params(opt_state)
        batch_in = batch[0]  # shape (batch_size, n_steps, n_in)
        batch_labels = batch[1]
        if e is not None:
            (loss, (acc, f1, rec, prec)), grads = update_opt_jit(w, batch_in, hyperparams, batch_labels)
            opt_state = opt_update(e, grads, opt_state)
        else:
            loss, (acc, f1, rec, prec) = loss_fn_jit(w, batch_in, hyperparams, batch_labels)
        local_loss.append(loss)
        local_acc.append(acc)
        local_prec.append(prec)
        local_rec.append(rec)
        local_f1.append(f1)
    mean_local_loss = jnp.mean(jnp.array(local_loss))
    mean_local_acc = jnp.mean(jnp.array(local_acc))
    mean_local_prec = jnp.mean(jnp.array(local_prec))
    mean_local_rec = jnp.mean(jnp.array(local_rec))
    mean_local_f1 = jnp.mean(jnp.array(local_f1))

    return key, opt_state, mean_local_loss, mean_local_acc, mean_local_prec, mean_local_rec, mean_local_f1