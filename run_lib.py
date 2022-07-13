# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools
from flax.metrics import tensorboard
# import wandb
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import ncsnpp
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import bound_likelihood
import sde_lib
from absl import flags

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  rng = jax.random.PRNGKey(config.seed)
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  if jax.host_id() == 0:
    writer = tensorboard.SummaryWriter(tb_dir)
    # wandb.init(project='score_sde', name=os.path.basename(os.path.normpath(workdir)),
    #          config=config.to_dict(), resume=True)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'linearvesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales,
                        linear=True)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  sampling_eps = config.sampling.smallest_time

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(checkpoint_meta_dir)
  # Resume training when intermediate checkpoints are detected
  state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  rng = state.rng

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  importance_weighting = config.training.importance_weighting
  smallest_time = config.training.smallest_time
  score_matching_order = config.training.score_matching_order

  train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting,
                                     importance_weighting=importance_weighting,
                                     smallest_time=smallest_time,
                                     score_matching_order=score_matching_order)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
  eval_step_fn = losses.get_step_fn(sde, score_model, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting,
                                    importance_weighting=importance_weighting,
                                    smallest_time=smallest_time,
                                    score_matching_order=score_matching_order)
  # Pmap (and jit-compile) multiple evaluation steps together for faster running
  p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size // jax.local_device_count(), config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Replicate the training state to run on multiple devices
  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.host_id() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.host_id())

  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    # Execute one training step
    (_, pstate), (ploss, ploss1, ploss2, ploss3) = p_train_step((next_rng, pstate), batch)
    loss = flax.jax_utils.unreplicate(ploss)
    loss1 = flax.jax_utils.unreplicate(ploss1)
    loss2 = flax.jax_utils.unreplicate(ploss2)
    loss3 = flax.jax_utils.unreplicate(ploss3)
    # Log to console, file and tensorboard on host 0
    if jax.host_id() == 0 and step % config.training.log_freq == 0:
      logging.info(
        "step: %d, training_loss: %.5e, training_loss1: %.5e, training_loss2: %.5e, training_loss3: %.5e" % (step, loss.mean(), loss1.mean(), loss2.mean(), loss3.mean())
      )
      # wandb.log({'training_loss': float(loss.mean())}, step=step)
      writer.scalar("training_loss", loss.mean(), step=step)
      writer.scalar("training_loss1", loss1.mean(), step=step)
      writer.scalar("training_loss2", loss2.mean(), step=step)
      writer.scalar("training_loss3", loss3.mean(), step=step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(rng=rng)
      checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                  step=step // config.training.snapshot_freq_for_preemption,
                                  keep=1, overwrite=True)
      del saved_state

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, _), (peval_loss, peval_loss1, peval_loss2, peval_loss3) = p_eval_step((next_rng, pstate), eval_batch)
      eval_loss = flax.jax_utils.unreplicate(peval_loss)
      eval_loss1 = flax.jax_utils.unreplicate(peval_loss1)
      eval_loss2 = flax.jax_utils.unreplicate(peval_loss2)
      eval_loss3 = flax.jax_utils.unreplicate(peval_loss3)
      if jax.host_id() == 0:
        logging.info(
          "step: %d, eval_loss: %.5e, eval_loss1: %.5e, eval_loss2: %.5e, eval_loss3: %.5e" % (step, eval_loss.mean(), eval_loss1.mean(), eval_loss2.mean(), eval_loss3.mean())
        )
        # wandb.log({'eval_loss': float(eval_loss.mean())}, step=step)
        writer.scalar("eval_loss", eval_loss.mean(), step=step)
        writer.scalar("eval_loss1", eval_loss1.mean(), step=step)
        writer.scalar("eval_loss2", eval_loss2.mean(), step=step)
        writer.scalar("eval_loss3", eval_loss3.mean(), step=step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng)
        checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                    step=step // config.training.snapshot_freq,
                                    keep=np.inf, overwrite=True)
        del saved_state

      # Generate and save samples
      if config.training.snapshot_sampling:
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
        sample, n = sampling_fn(sample_rng, pstate)
        this_sample_dir = os.path.join(
          sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
        tf.io.gfile.makedirs(this_sample_dir)
        image_grid = sample.reshape((-1, *sample.shape[2:]))
        nrow = int(np.sqrt(image_grid.shape[0]))
        sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          utils.save_image(image_grid, fout, nrow=nrow, padding=2)


def evaluate(config,
             workdir,
             eval_folder="eval",
             deq_folder="flowpp_dequantizer"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  if config.eval.enable_loss or config.eval.enable_bpd:
    # Build data pipeline
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=1,
                                                uniform_dequantization=config.data.uniform_dequantization,
                                                evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'linearvesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales,
                        linear=True)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  sampling_eps = config.sampling.smallest_time

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting
    importance_weighting = config.training.importance_weighting
    smallest_time = config.training.smallest_time
    score_matching_order = config.training.score_matching_order

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, score_model,
                                   train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous, likelihood_weighting=likelihood_weighting,
                                   importance_weighting=importance_weighting,
                                   smallest_time=smallest_time,
                                   score_matching_order=score_matching_order)
    # Pmap (and jit-compile) multiple evaluation steps together for faster execution
    p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step), axis_name='batch', donate_argnums=1)

  if config.eval.enable_bpd:
    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    if config.eval.dequantizer:
      train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                          additional_dim=None,
                                                          uniform_dequantization=False,
                                                          evaluation=True)  # For data-dependent initialization. Must take values in [0, 1]
      init_data = jnp.asarray(next(iter(train_ds_bpd))['image']._numpy())
      rng, step_rng = jax.random.split(rng)
      deq_model, deq_init_params = mutils.data_dependent_init_of_dequantizer(step_rng, config, init_data)
      deq_optimizer = losses.get_optimizer(config).create(deq_init_params)
      deq_state = mutils.DeqState(step=0, optimizer=deq_optimizer,
                                  lr=config.optim.lr, ema_rate=config.deq.ema_rate,
                                  params_ema=deq_init_params, ema_train_bpd=0,
                                  ema_eval_bpd=0, rng=rng)
      deq_state = checkpoints.restore_checkpoint(os.path.join(workdir, deq_folder, "checkpoints"),
                                                 deq_state, step=6)
      # deq_state = checkpoints.restore_checkpoint(os.path.join(workdir, deq_folder, "checkpoints"),
      #                                            deq_state, step=4)
      logging.info("Successfully loaded the variational dequantizer!")
      dequantizer = mutils.get_dequantizer(deq_model, deq_state.params_ema, train=False)
      p_dequantizer = jax.pmap(dequantizer, axis_name='batch')

    else:
      train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                          additional_dim=None,
                                                          uniform_dequantization=True, evaluation=True)

    if config.eval.bpd_dataset.lower() == 'train':
      ds_bpd = train_ds_bpd
      bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
      # Go over the dataset 5 times when computing likelihood on the test dataset
      ds_bpd = eval_ds_bpd
      bpd_num_repeats = config.eval.num_repeats
    else:
      raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    if config.eval.bound:
      likelihood_fn = bound_likelihood.get_likelihood_bound_fn(sde, score_model, inverse_scaler,
                                                               dsm=config.eval.dsm,
                                                               eps=config.training.smallest_time,
                                                               importance_weighting=True,
                                                               N=1000,
                                                               eps_offset=config.eval.offset)
    else:
      likelihood_fn = likelihood.get_likelihood_fn(sde, score_model, inverse_scaler, eps=config.training.smallest_time)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                      config.data.image_size, config.data.image_size,
                      config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Create different random states for different hosts in a multi-host environment (e.g., TPU pods)
  rng = jax.random.fold_in(rng, jax.host_id())

  # A data class for storing intermediate results to resume evaluation after pre-emption
  @flax.struct.dataclass
  class EvalMeta:
    ckpt_id: int
    sampling_round_id: int
    bpd_round_id: int
    rng: Any

  # Add one additional round to get the exact number of samples as required.
  num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
  if config.eval.enable_bpd:
    num_bpd_rounds = len(ds_bpd) * bpd_num_repeats
  else:
    num_bpd_rounds = 1

  # Restore evaluation after pre-emption
  eval_meta = EvalMeta(ckpt_id=config.eval.begin_ckpt, sampling_round_id=-1, bpd_round_id=-1, rng=rng)
  eval_meta = checkpoints.restore_checkpoint(
    eval_dir, eval_meta, step=None, prefix=f"meta_{jax.host_id()}_")

  if eval_meta.bpd_round_id < num_bpd_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = eval_meta.bpd_round_id + 1
    begin_sampling_round = 0

  elif eval_meta.sampling_round_id < num_sampling_rounds - 1:
    begin_ckpt = eval_meta.ckpt_id
    begin_bpd_round = num_bpd_rounds
    begin_sampling_round = eval_meta.sampling_round_id + 1

  else:
    begin_ckpt = eval_meta.ckpt_id + 1
    begin_bpd_round = 0
    begin_sampling_round = 0

  rng = eval_meta.rng

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
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

    # Replicate the training state for executing on multiple devices
    pstate = flax.jax_utils.replicate(state)
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)  # pylint: disable=protected-access
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        (_, _), p_eval_loss = p_eval_step((next_rng, pstate), eval_batch)
        eval_loss = flax.jax_utils.unreplicate(p_eval_loss)
        all_losses.extend(eval_loss)
        if (i + 1) % 1000 == 0 and jax.host_id() == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = jnp.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      begin_repeat_id = begin_bpd_round // len(ds_bpd)
      begin_batch_id = begin_bpd_round % len(ds_bpd)
      # Repeat multiple times to reduce variance when needed
      for repeat in range(begin_repeat_id, bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for _ in range(begin_batch_id):
          next(bpd_iter)
        for batch_id in range(begin_batch_id, len(ds_bpd)):
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          if tf.io.gfile.exists(os.path.join(eval_dir,
                                             f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz")):
            continue
          batch = next(bpd_iter)
          eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
          if config.eval.dequantizer:
            rng, step_rng = jax.random.split(rng)
            data = eval_batch['image']
            u = jax.random.normal(step_rng, data.shape)
            noise, logpd = p_dequantizer(u, inverse_scaler(data))
            data = scaler((inverse_scaler(data) * 255. + noise) / 256.)
            bpd_d = -logpd / np.log(2.)
            dim = np.prod(noise.shape[2:])
            bpd_d = bpd_d / dim
          else:
            data = eval_batch['image']

          rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
          step_rng = jnp.asarray(step_rng)
          bpd = likelihood_fn(step_rng, pstate, data)[0]
          if config.eval.dequantizer:
            bpd = bpd + bpd_d
          bpd = bpd.reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d / %d, mean bpd: %6f" % (ckpt, repeat, batch_id, len(ds_bpd), jnp.mean(jnp.asarray(bpds))))
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

          eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=bpd_round_id, rng=rng)
          # Save intermediate states to resume evaluation after pre-emption
          checkpoints.save_checkpoint(
            eval_dir,
            eval_meta,
            step=ckpt * (num_sampling_rounds + num_bpd_rounds) + bpd_round_id,
            keep=1,
            prefix=f"meta_{jax.host_id()}_", overwrite=True)
    else:
      # Skip likelihood computation and save intermediate states for pre-emption
      eval_meta = eval_meta.replace(ckpt_id=ckpt, bpd_round_id=num_bpd_rounds - 1)
      checkpoints.save_checkpoint(
        eval_dir,
        eval_meta,
        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_bpd_rounds - 1,
        keep=1,
        prefix=f"meta_{jax.host_id()}_", overwrite=True)

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      state = jax.device_put(state)
      # Run sample generation for multiple rounds to create enough samples
      # Designed to be pre-emption safe. Automatically resumes when interrupted
      for r in range(begin_sampling_round, num_sampling_rounds):
        if jax.host_id() == 0:
          logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(
          eval_dir, f"ckpt_{ckpt}_host_{jax.host_id()}")
        tf.io.gfile.makedirs(this_sample_dir)

        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
        samples, n = sampling_fn(sample_rng, pstate)
        samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        if not inceptionv3:
          # Force garbage collection before calling TensorFlow code for Inception network
          gc.collect()
          latents = evaluation.run_inception_distributed(samples, inception_model,
                                                         inceptionv3=inceptionv3)
          # Force garbage collection again before returning to JAX code
          gc.collect()
          # Save latent represents of the Inception network to disk or Google Cloud Storage
          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(
              io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
            fout.write(io_buffer.getvalue())

        # Update the intermediate evaluation state
        eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=r, rng=rng)
        # Save an intermediate checkpoint directly if not the last round.
        # Otherwise save eval_meta after computing the Inception scores and FIDs
        if r < num_sampling_rounds - 1:
          checkpoints.save_checkpoint(
            eval_dir,
            eval_meta,
            step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
            keep=1,
            prefix=f"meta_{jax.host_id()}_", overwrite=True)

      # Compute inception scores, FIDs and KIDs.
      if jax.host_id() == 0 and not inceptionv3:
        # Load all statistics that have been previously computed and saved for each host
        all_logits = []
        all_pools = []
        for host in range(jax.host_count()):
          this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{host}")

          stats = tf.io.gfile.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
          wait_message = False
          while len(stats) < num_sampling_rounds:
            if not wait_message:
              logging.warning("Waiting for statistics on host %d" % (host,))
              wait_message = True
            stats = tf.io.gfile.glob(
              os.path.join(this_sample_dir, "statistics_*.npz"))
            time.sleep(30)

          for stat_file in stats:
            with tf.io.gfile.GFile(stat_file, "rb") as fin:
              stat = np.load(fin)
              if not inceptionv3:
                all_logits.append(stat["logits"])
              all_pools.append(stat["pool_3"])

        if not inceptionv3:
          all_logits = np.concatenate(
            all_logits, axis=0)[:config.eval.num_samples]
        all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

        # Load pre-computed dataset statistics.
        data_stats = evaluation.load_dataset_stats(config)
        data_pools = data_stats["pool_3"]

        # Compute FID/KID/IS on all samples together.
        if not inceptionv3:
          inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
        else:
          inception_score = -1

        fid = tfgan.eval.frechet_classifier_distance_from_activations(
          data_pools, all_pools)
        # Hack to get tfgan KID work for eager execution.
        tf_data_pools = tf.convert_to_tensor(data_pools)
        tf_all_pools = tf.convert_to_tensor(all_pools)
        kid = tfgan.eval.kernel_classifier_distance_from_activations(
          tf_data_pools, tf_all_pools).numpy()
        del tf_data_pools, tf_all_pools

        logging.info(
          "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
            ckpt, inception_score, fid, kid))

        with tf.io.gfile.GFile(os.path.join(eval_dir, f"report_{ckpt}.npz"),
                               "wb") as f:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
          f.write(io_buffer.getvalue())
      elif not inceptionv3:
        # For host_id() != 0.
        # Use file existence to emulate synchronization across hosts
        while not tf.io.gfile.exists(os.path.join(eval_dir, f"report_{ckpt}.npz")):
          time.sleep(1.)

      # Save eval_meta after computing IS/KID/FID to mark the end of evaluation for this checkpoint
      checkpoints.save_checkpoint(
        eval_dir,
        eval_meta,
        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + r + num_bpd_rounds,
        keep=1,
        prefix=f"meta_{jax.host_id()}_", overwrite=True)

    else:
      # Skip sampling and save intermediate evaluation states for pre-emption
      eval_meta = eval_meta.replace(ckpt_id=ckpt, sampling_round_id=num_sampling_rounds - 1, rng=rng)
      checkpoints.save_checkpoint(
        eval_dir,
        eval_meta,
        step=ckpt * (num_sampling_rounds + num_bpd_rounds) + num_sampling_rounds - 1 + num_bpd_rounds,
        keep=1,
        prefix=f"meta_{jax.host_id()}_", overwrite=True)

    begin_bpd_round = 0
    begin_sampling_round = 0

  # Remove all meta files after finishing evaluation
  meta_files = tf.io.gfile.glob(
    os.path.join(eval_dir, f"meta_{jax.host_id()}_*"))
  for file in meta_files:
    tf.io.gfile.remove(file)


def train_deq(config, workdir, deq_workdir):
  rng = jax.random.PRNGKey(config.seed)
  tb_dir = os.path.join(deq_workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  if jax.host_id() == 0:
    writer = tensorboard.SummaryWriter(tb_dir)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
  elif config.training.sde.lower() == 'linearvesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales,
                        linear=True)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=False)
  # train_ds, eval_ds, _ = datasets.get_dataset(config,
  #                                             additional_dim=None,
  #                                             uniform_dequantization=False)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # For data-dependent initialization. Must take values in [0, 1]
  init_data = jnp.asarray(next(train_iter)['image']._numpy())[:, 0, ...]
  # init_data = jnp.asarray(next(train_iter)['image']._numpy())
  rng, step_rng = jax.random.split(rng)
  deq_model, initial_params = mutils.data_dependent_init_of_dequantizer(step_rng, config, init_data)
  optimizer = losses.get_optimizer(config).create(initial_params)

  state = mutils.DeqState(step=0, optimizer=optimizer, lr=config.optim.lr,
                          ema_rate=config.deq.ema_rate,
                          params_ema=initial_params,
                          ema_train_bpd=0,
                          ema_eval_bpd=0,
                          rng=rng)  # pytype: disable=wrong-keyword-args

  # Create checkpoints directory
  checkpoint_dir = os.path.join(deq_workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(deq_workdir, "checkpoints-meta")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(checkpoint_meta_dir)
  # Resume training when intermediate checkpoints are detected
  state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  rng = state.rng
  ema_train_bpd = state.ema_train_bpd
  ema_eval_bpd = state.ema_eval_bpd

  # Load score model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  score_optimizer = losses.get_optimizer(config).create(initial_params)
  score_state = mutils.State(step=0, optimizer=score_optimizer,
                             lr=config.optim.lr,
                             model_state=init_model_state,
                             ema_rate=config.model.ema_rate,
                             params_ema=initial_params,
                             rng=rng)  # pytype: disable=wrong-keyword-args

  ckpt_filename = os.path.join(os.path.join(workdir, "checkpoints"), f"checkpoint_{config.eval.ckpt_id}")
  assert tf.io.gfile.exists(ckpt_filename)
  score_state = checkpoints.restore_checkpoint(os.path.join(workdir, "checkpoints"), score_state,
                                               step=config.eval.ckpt_id)
  score_fn = mutils.get_score_fn(sde, score_model, score_state.params_ema,
                                 score_state.model_state, train=False,
                                 continuous=True, return_state=False)
  logging.info("Successfully loaded the score model!")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  smallest_time = config.training.smallest_time
  deq_offset = config.deq.offset

  train_step_fn = losses.get_dequantizer_step_fn(sde, score_fn, deq_model, scaler, inverse_scaler,
                                                 train=True, optimize_fn=optimize_fn,
                                                 importance_weighting=True,
                                                 smallest_time=smallest_time,
                                                 eps_offset=deq_offset)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
  # p_train_step = jax.pmap(train_step_fn, axis_name='batch')
  eval_step_fn = losses.get_dequantizer_step_fn(sde, score_fn, deq_model, scaler, inverse_scaler,
                                                train=False, optimize_fn=optimize_fn,
                                                importance_weighting=True,
                                                smallest_time=smallest_time,
                                                eps_offset=deq_offset)
  # Pmap (and jit-compile) multiple evaluation steps together for faster running
  p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)
  # p_eval_step = jax.pmap(eval_step_fn, axis_name='batch')

  # Replicate the training state to run on multiple devices
  pstate = flax_utils.replicate(state)
  num_train_steps = config.deq.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.host_id() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.host_id())

  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    # Execute one training step
    (_, pstate), ploss = p_train_step((next_rng, pstate), batch)
    loss = flax.jax_utils.unreplicate(ploss)
    # Log to console, file and tensorboard on host 0
    if jax.host_id() == 0 and step % config.training.log_freq == 0:
      loss = jax.tree_map(lambda x: x.mean(), loss)
      ema_train_bpd = 0.99 * ema_train_bpd + 0.01 * loss[1]
      logging.info(
        f"step: {step}, training_loss: {loss[0]:.5e}, training_bpd: {loss[1]:.5e}, ema_bpd: {ema_train_bpd:.5e}")
      writer.scalar('training_loss', loss[0], step=step)
      writer.scalar('training_bpd', loss[1], step=step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(rng=rng, ema_train_bpd=ema_train_bpd, ema_eval_bpd=ema_eval_bpd)
      checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                  step=step // config.training.snapshot_freq_for_preemption,
                                  keep=1, overwrite=True)
      del saved_state

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)
      eval_loss = flax.jax_utils.unreplicate(peval_loss)
      if jax.host_id() == 0:
        eval_loss = jax.tree_map(lambda x: x.mean(), eval_loss)
        ema_eval_bpd = 0.9 * ema_eval_bpd + 0.1 * eval_loss[1]
        logging.info(
          f"step: {step}, eval_loss: {eval_loss[0]:.5e}, eval_bpd: {eval_loss[1]:.5e}, ema_bpd: {ema_eval_bpd:.5e}")
        writer.scalar('eval_loss', eval_loss[0], step=step)
        writer.scalar('eval_bpd', eval_loss[1], step=step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng, ema_train_bpd=ema_train_bpd, ema_eval_bpd=ema_eval_bpd)
        checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                    step=step // config.training.snapshot_freq,
                                    keep=np.inf, overwrite=True)
        del saved_state