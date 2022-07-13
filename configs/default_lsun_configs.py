import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 50
  training.n_iters = 3400001
  training.snapshot_freq = 50000
  training.log_freq = 100
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.n_jitted_steps = 1
  training.reduce_mean = False
  training.smallest_time = 1e-5
  training.score_matching_order = 1

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075
  sampling.smallest_time = 1e-3

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  # evaluate.begin_ckpt = 126
  # evaluate.end_ckpt = 126
  evaluate.begin_ckpt = 52
  evaluate.end_ckpt = 52
  evaluate.batch_size = 100
  evaluate.enable_sampling = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = False
  evaluate.enable_bpd = True
  evaluate.bpd_dataset = 'test'
  evaluate.num_repeats = 5
  evaluate.bound = False
  evaluate.dsm = True
  evaluate.dequantizer = False
  evaluate.offset = True

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'LSUN'
  data.image_size = 256
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'
  model.data_init = False
  model.trainable_embedding = False

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
