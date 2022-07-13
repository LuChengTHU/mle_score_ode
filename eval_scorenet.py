from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os
import jax
import datasets
import sde_lib
from models import utils as mutils
import losses
from flax.training import checkpoints
import logging
import time
import jax.numpy as jnp
from utils import batch_mul
from scipy import integrate
from models import ncsnpp
import numpy as np
import flax

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])

NUM_TIMEPOINTS = 100


def batch_div(a, b):
    return jax.vmap(lambda a, b: a / b)(a, b)


def get_eval_fn(sde, model, vp, hutchinson_type="Rademacher", rtol=1e-5, atol=1e-5, method="RK45", eps=1e-5, nt=2):
    def drift_fn(state, x, t):
        score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
        drift, diffusion = sde.sde(x, t)
        score = score_fn(x, t)
        drift = drift - batch_mul(diffusion ** 2, score * 0.5)
        return drift

    @jax.pmap
    def get_model_grad_div_jvp(state, x, t, eps, v):
        def value_div_jvp_fn(x):
            fn = lambda inputs: drift_fn(state, inputs, t)
            drift, drift_jvp = jax.jvp(fn, (x,), (eps,))
            return jnp.sum(drift_jvp * eps + drift * v), drift

        grad_div_jvp, drift = jax.grad(value_div_jvp_fn, has_aux=True)(x)
        return jax.lax.stop_gradient(drift), jax.lax.stop_gradient(grad_div_jvp)

    @jax.pmap
    def get_score_grad_div(state, x, t, eps):
        def value_div_fn(x):
            score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
            fn = lambda inputs: score_fn(inputs, t)
            score, score_jvp = jax.jvp(fn, (x,), (eps,))
            div = jnp.sum((score_jvp * eps).reshape((x.shape[0], -1)), axis=-1, keepdims=True)
            return jnp.sum(div), (score, div)

        grad_div, (score, div) = jax.grad(value_div_fn, has_aux=True)(x)
        return jax.lax.stop_gradient(score), jax.lax.stop_gradient(div), jax.lax.stop_gradient(grad_div)

    @jax.pmap
    def prior_score(z):
        if vp:
            return -z
        else:
            return -z / sde.sigma_max ** 2

    @jax.pmap
    def norm_2(x):
        return jnp.sum(jnp.square(x).reshape((x.shape[0], -1)), -1)

    p_sde = jax.pmap(sde.sde)
    p_marginal_prob = jax.pmap(sde.marginal_prob)
    p_batch_div = jax.pmap(batch_div)
    p_drift_fn = jax.pmap(drift_fn)

    def eval_fn(prng, pstate, data):
        rng, step_rng = jax.random.split(flax.jax_utils.unreplicate(prng))

        # shape = [num_devices, batch_per_device, 32, 32, 3]
        shape = data.shape
        dim = np.prod(shape[2:])
        ts = np.linspace(eps, sde.T, nt)

        if hutchinson_type == "Gaussian":
            epsilon = jax.random.normal(step_rng, shape)
        elif hutchinson_type == "Rademacher":
            epsilon = jax.random.rademacher(step_rng, shape, dtype=jnp.float32)
        else:
            raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

        def ode_func_forward(t, x):
            sample = mutils.from_flattened_numpy(x, shape)
            vec_t = jnp.ones((shape[0], shape[1])) * t
            drift = p_drift_fn(pstate, sample, vec_t)
            drift = mutils.to_flattened_numpy(drift)
            return drift

        def ode_func_backward(t, x):
            sample, score = np.split(x, 2)
            sample = mutils.from_flattened_numpy(sample, shape)
            score = mutils.from_flattened_numpy(score, shape)
            vec_t = jnp.ones((shape[0], shape[1])) * t
            drift, grad_div_jvp = get_model_grad_div_jvp(pstate, sample, vec_t, epsilon, score)
            drift = mutils.to_flattened_numpy(drift)
            grad_div_jvp = mutils.to_flattened_numpy(grad_div_jvp)
            return np.concatenate([drift, -grad_div_jvp], axis=0)

        init = mutils.to_flattened_numpy(data)
        solution = integrate.solve_ivp(ode_func_forward, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
        xT = mutils.from_flattened_numpy(solution.y[:, -1], shape)
        score_T = prior_score(xT)
        init = np.concatenate([mutils.to_flattened_numpy(xT), mutils.to_flattened_numpy(score_T)], axis=0)
        solution = integrate.solve_ivp(ode_func_backward, (sde.T, eps), init, rtol=rtol, atol=atol, method=method, t_eval=np.flip(ts))
        y = np.flip(solution.y, axis=1)
        s_with_odes, s_with_normals, ode_with_normals, s_div_with_normals, s_grad_div_with_normals = [], [], [], [], []
        for i, t in enumerate(ts):
            ss = jnp.asarray(y[:, i])
            xt, ode_score = jnp.split(ss, 2)
            xt = mutils.from_flattened_numpy(xt, shape)
            ode_score = mutils.from_flattened_numpy(ode_score, shape)
            vec_t = jnp.ones((shape[0], shape[1])) * t
            _, diffusion = p_sde(xt, vec_t)
            gt2 = diffusion ** 2
            score, score_div, score_grad_div = get_score_grad_div(pstate, xt, vec_t, epsilon)
            _, std = p_marginal_prob(xt, vec_t)
            std2 = jnp.expand_dims(std, 2) ** 2
            score_normal = -p_batch_div(xt, std2)
            score_normal_div = -dim / std2
            score_normal_grad_div = 0.0
            s_with_odes.append(jnp.mean(gt2 * norm_2(ode_score - score)))
            s_with_normals.append(jnp.mean(gt2 * norm_2(score - score_normal)))
            ode_with_normals.append(jnp.mean(gt2 * norm_2(ode_score - score_normal)))
            s_div_with_normals.append(jnp.mean(gt2 * norm_2(score_div - score_normal_div)))
            s_grad_div_with_normals.append(jnp.mean(gt2 * norm_2(score_grad_div - score_normal_grad_div)))
        return (
            jnp.asarray(s_with_odes),
            jnp.asarray(s_with_normals),
            jnp.asarray(ode_with_normals),
            jnp.asarray(s_div_with_normals),
            jnp.asarray(s_grad_div_with_normals),
        )

    return eval_fn


def evaluate_scorenet(config, workdir):
    rng = jax.random.PRNGKey(config.seed + 1)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    eval_filename = os.path.join(workdir, "eval.npz")

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        vp = True
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        vp = False
    elif config.training.sde.lower() == "linearvesde":
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, linear=True)
        vp = False
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Initialize model
    rng, model_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
    optimizer = losses.get_optimizer(config).create(initial_params)
    state = mutils.State(
        step=0,
        optimizer=optimizer,
        lr=config.optim.lr,
        model_state=init_model_state,
        ema_rate=config.model.ema_rate,
        params_ema=initial_params,
        rng=rng,
    )  # pytype: disable=wrong-keyword-args

    train_ds, eval_ds, _ = datasets.get_dataset(config, additional_dim=None, uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == "train":
        ds = train_ds
        num_repeats = 1
    elif config.eval.bpd_dataset.lower() == "test":
        # Go over the dataset 5 times when computing likelihood on the test dataset
        ds = eval_ds
        num_repeats = config.eval.num_repeats
    else:
        raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
    rng = jax.random.fold_in(rng, jax.host_id())
    nt = NUM_TIMEPOINTS
    ts = np.linspace(config.training.smallest_time, sde.T, nt)
    for ckpt in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1):
        logging.info("testing checkpoint: %d" % (ckpt,))
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}".format(ckpt))
        while not tf.io.gfile.exists(ckpt_filename):
            if not waiting_message_printed and jax.host_id() == 0:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        try:
            state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
        except:
            time.sleep(60)
            try:
                state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)
            except:
                time.sleep(120)
                state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=ckpt)

        eval_fn = get_eval_fn(sde, score_model, vp, eps=config.training.smallest_time, nt=nt)
        # Replicate the training state for executing on multiple devices
        pstate = flax.jax_utils.replicate(state)

        # Compute the loss function on the full evaluation dataset if loss computation is enabled
        s_with_odes, s_with_normals, ode_with_normals, s_div_with_normals, s_grad_div_with_normals = [], [], [], [], []
        # Repeat multiple times to reduce variance when needed
        for repeat in range(num_repeats):
            ds_iter = iter(ds)  # pytype: disable=wrong-arg-types
            for batch_id in range(len(ds)):
                batch = next(ds_iter)
                eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
                data = eval_batch["image"]
                rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
                step_rng = jnp.asarray(step_rng)
                s_with_ode, s_with_normal, ode_with_normal, s_div_with_normal, s_grad_div_with_normal = eval_fn(step_rng, pstate, data)
                s_with_odes.append(s_with_ode)
                s_with_normals.append(s_with_normal)
                ode_with_normals.append(ode_with_normal)
                s_div_with_normals.append(s_div_with_normal)
                s_grad_div_with_normals.append(s_grad_div_with_normal)
                s_with_odes_mean, s_with_normals_mean, ode_with_normals_mean, s_div_with_normals_mean, s_grad_div_with_normals_mean = (
                    jnp.mean(jnp.asarray(s_with_odes), axis=0),
                    jnp.mean(jnp.asarray(s_with_normals), axis=0),
                    jnp.mean(jnp.asarray(ode_with_normals), axis=0),
                    jnp.mean(jnp.asarray(s_div_with_normals), axis=0),
                    jnp.mean(jnp.asarray(s_grad_div_with_normals), axis=0),
                )
                np.savez_compressed(
                    eval_filename,
                    s_with_odes=np.asarray(s_with_odes_mean),
                    s_with_normals=np.asarray(s_with_normals_mean),
                    ode_with_normals=np.asarray(ode_with_normals_mean),
                    s_div_with_normals=np.asarray(s_div_with_normals_mean),
                    s_grad_div_with_normals=np.asarray(s_grad_div_with_normals_mean),
                )
                for i, t in enumerate(ts):
                    if i % 10 == 0:
                        logging.info(
                            "ckpt: %d, repeat: %d/%d, batch: %d/%d, t: %6f, s/ode: %6f, s/normal: %6f, ode/normal: %6f, s div/normal: %6f, s grad div/normal: %6f"
                            % (
                                ckpt,
                                repeat,
                                num_repeats,
                                batch_id,
                                len(ds),
                                t,
                                s_with_odes_mean[i],
                                s_with_normals_mean[i],
                                ode_with_normals_mean[i],
                                s_div_with_normals_mean[i],
                                s_grad_div_with_normals_mean[i],
                            )
                        )


def main(argv):
    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    evaluate_scorenet(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    app.run(main)
