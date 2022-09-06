# Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching (ICML 2022)

The official code for the paper [Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching](https://arxiv.org/abs/2206.08265) by Cheng Lu, Kaiwen Zheng, Fan Bao, Jianfei Chen, Chongxuan Li and Jun Zhu, published in ICML 2022.

The code implementation is based on [score_flow](https://github.com/yang-song/score_flow) by Yang Song.

--------------------

Score-based diffusion models include two types: ScoreSDE and ScoreODE. [Previous work](https://arxiv.org/abs/2101.09258) showed that the weighted combination of first-order score matching losses can upper bound the Kullback–Leibler divergence between the data distribution and the ScoreSDE model distribution. However, the relationship between score matching and ScoreODE is unclear. In this work, we prove that:

- The model distributions of ScoreSDE and ScoreODE are **always different** if the data distribution is not a Gaussian distribution.
- To upper bound the KL-divergence of ScoreODE, we need first-order, second-order and third-order score matching for the score model.
- We further propose an error-bounded high-order denoising score matching method. The higher-order score matching error can be exactly upper bounded by the training error and the lower-order score matching errors, which enables  high-order score matching.

In short, The previous work [Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258) is a method for maximum likelihood training of **ScoreSDE** (a.k.a. **diffusion SDE**), and our work is a method for maximum likelihood training of **ScoreODE** (a.k.a. **diffusion ODE**). 
 
## Code Structure
The code implementation is based on [score_flow](https://github.com/yang-song/score_flow) by Yang Song. We further implement the proposed high-order denoising score matching losses in `losses.py`.

## How to run the code

### Dependencies

We use the same denpendencies as [score_flow](https://github.com/yang-song/score_flow). To install the packages, we recommend the ``jaxlib==0.1.69``. You need to find a corresponding version for your python3 version and cuda version at: [https://storage.googleapis.com/jax-releases/jax_cuda_releases.html](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html). For example, to install ``jaxlib==0.1.69`` for `python==3.7` and `cuda==11.1`, you need to firstly download the wheel file:
```sh
wget https://storage.googleapis.com/jax-releases/cuda111/jaxlib-0.1.69+cuda111-cp37-none-manylinux2010_x86_64.whl
```
and then run the following command to install `jaxlib`:
```sh
pip3 install jaxlib-0.1.69+cuda111-cp37-none-manylinux2010_x86_64.whl
```
After install `jaxlib`, you need to run to following command to install the other packages:
```sh
pip3 install -r requirements.txt
```

### Stats files for quantitative evaluation

We use the same stats files by [score_flow](https://github.com/yang-song/score_flow) for computing FID and Inception scores for CIFAR-10 and ImageNet 32x32. You can find `cifar10_stats.npz` and `imagenet32_stats.npz` under the directory `assets/stats` in Yang Song's [Google drive](https://drive.google.com/drive/folders/1gbDrVrFVSupFMRoK7HZo8aFgPvOtpmqB?usp=sharing). Download them and save to `assets/stats/` in the code repo.

### Usage
The running command is the same as [score_flow](https://github.com/yang-song/score_flow). Here are some common options:

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval>: Running mode: train or eval. We did not train our model by further variational dequantizations.
  --workdir: Working directory
```

* `config` is the path to the config file. Our config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

  **Naming conventions of config files**: the name of a config file contains the following attributes:

  * dataset: Either `cifar10` or `imagenet32`
  * model: Either `ddpmpp_continuous` or `ddpmpp_deep_continuous`

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for supporting pre-emption recovery, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta` . When set to "eval", it can do the following:

  * Compute the log-likelihood on the training or test dataset.
  * Compute the lower bound of the log-likelihood on the training or test dataset.
  * Evaluate the loss function on the test / validation dataset.  
  * Generate a fixed number of samples and compute its Inception score, FID, or KID. Prior to evaluation, stats files must have already been downloaded/computed and stored in `assets/stats`.
	
These functionalities can be configured through config files, or more conveniently, through the command-line support of the `ml_collections` package. 

### Configurations for high-order denoising score matching
To set the order of the score matching training losses, set `--config.training.score_matching_order` to be `1` (the previous first-order) or `2` or `3`. Note that for third-order score matching training, the batch size needs to turn smaller to avoid OOM.

### Configurations for evaluation
To generate samples and evaluate sample quality, use the  `--config.eval.enable_sampling` flag; to compute log-likelihoods, use the `--config.eval.enable_bpd` flag, and specify `--config.eval.dataset=train/test` to indicate whether to compute the likelihoods on the training or test dataset. Turn on `--config.eval.bound` to evaluate the variational bound for the log-likelihood. Enable `--config.eval.dequantizer` to use variational dequantization for likelihood computation. `--config.eval.num_repeats` configures the number of repetitions across the dataset (more can reduce the variance of the likelihoods; default to 5).

## Pretrained checkpoints
The pretrained checkpoints can be found in the [Released](https://github.com/LuChengTHU/mle_score_ode/releases) page.

## Train high-order DSM by pretrained checkpoints
For VESDE on CIFAR-10, we use the pretrained checkpoints by first-order DSM in [score_sde checkpoints](https://drive.google.com/drive/folders/1RAG8qpOTURkrqXKwdAR1d6cU9rwoQYnH?usp=sharing).

For VESDE on ImageNet32, as score_sde did not provide the checkpoints, we train the first-order model by ourselves, and then train the model by the high-order DSM.

For VPSDE, we use the pretrained checkpoints by first-order DSM in [score_flow checkpoints](https://drive.google.com/drive/folders/1gbDrVrFVSupFMRoK7HZo8aFgPvOtpmqB?usp=sharing).

## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{lu2022maximum,
  title={Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching},
  author={Lu, Cheng and Zheng, Kaiwen and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  booktitle={International Conference on Machine Learning},
  year={2022}
  organization={PMLR}
}
```

This work is built upon some previous papers which might also interest you:

* Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "Score-Based Generative Modeling through Stochastic Differential Equations". *Proceedings of the 9th International Conference on Learning Representations*, 2021.
* Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. "Maximum Likelihood Training of Score-Based Diffusion Models". *Advances in Neural Information Processing Systems*, 2021.
