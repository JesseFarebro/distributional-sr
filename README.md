# Distributional Successor Measure

This repository contains the reference implementation of the Distributional Successor Measure presented in:

**[A Distributional Analogue to the Successor Representation](https://arxiv.org/abs/2402.08530)**

by [Harley Wiltzer](https://harwiltz.github.io/)* & [Jesse Farebrother](https://brosa.ca)*, [Arthur Gretton](https://www.gatsby.ucl.ac.uk/~gretton/), [Yunhao Tang](https://robintyh1.github.io/), [André Baretto](https://sites.google.com/view/andrebarreto/about), [Will Dabney](https://willdabney.com/), [Marc G. Bellemare](http://www.marcgbellemare.info/), and [Mark Rowland](https://sites.google.com/view/markrowland).

https://github.com/JesseFarebro/distributional-sr/assets/1377567/eea0a53a-65d7-4201-a234-6609d1166d11

The Distributional Successor Measure (DSM) a new approach for distributional reinforcement learning which proposes a clean separation of transition structure and reward in the learning process. Analogous to how the successor representation (SR) describes the expected consequences of behaving according to a given policy, our distributional successor measure describes the distributional consequences of this behaviour. This repository contains the code for learning a $\delta$-model, our proposed representation that learns the distributional SM.

## Setup

This project makes heavy use of [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), and [Fiddle](https://github.com/google/fiddle). We use [pdm](https://pdm-project.org/latest/) to manage our dependencies. With the lockfile `pdm.lock` you should be able to faithfully instantiate the same environment we used to train our $\delta$-models. To do so, you can run the following commands,
```sh
pdm venv create
pdm install
```

If you're looking to build upon this project you might want to read more about how to use the Fiddle configuration library here: [JesseFarebro/fiddle-demo](https://github.com/JesseFarebro/fiddle-demo).

## Generating the Fixed Datasets

The following command will train a policy on the desired environment before generating a dataset
of transitions from the learned policy. For example,

```sh
python -m sr.scripts.make_dataset --env Pendulum-v1 --dataset_path datasets/pendulum/sac/dataset.pkl --policy_path datasets/pendulum/sac/policy
```

NOTE: The policy will be cached and if you don't specify the `--force` flag it will skip the policy optimization step.

## Training a $\delta$-Model

To train the $\delta$-model from the paper you can simply run:

```sh
python -m dsm --workdir logdir
```

where `logdir` will store checkpoints of the saved model. Plots of the learned return distributions and various metrics will be logged periodically throughout training. These plots and metrics can be found in the experiment tracker (defaults to Aim).

### Experiment Tracking

You can switch how the experiment is logged either using Weights & Biases or Aim with the flag `--metric_writer {wandb, aim}`. Specific options for each of these methods can be configured via `--wandb.{save_code,tags,name,group,mode}` and `--aim.{repo=,experiment,log_system_params}` respectively.

To run the local Aim server you can simply run: `pdm run aim` and then navigate to the provided URL.

## Citation
If you build on our work or find it useful, please cite it using the following bibtex.

```bibtex
@article{wiltzer2024dsm,
    title={A Distributional Analogue to the Successor Representation},
    author={Harley Wiltzer and Jesse Farebrother and Arthur Gretton and Yunhao Tang and Andr\'e Barreto and Will Dabney and Marc G Bellemare and Mark Rowland},
    year={2024},
    journal={arXiv preprint arXiv:2402.08530},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
