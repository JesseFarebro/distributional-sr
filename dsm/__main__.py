import contextlib
import inspect
import logging
import operator
import os
import sys
import typing
from typing import Any

import fancyflags as ff
import fiddle as fdl
import fiddle.extensions.jax
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import tqdm.rich as tqdm
from absl import app, flags
from absl import logging as absl_logging
from clu import metric_writers
from dm_env import specs
from etils import epath
from fiddle import absl_flags as fdl_flags
from fiddle import printing
from fiddle.codegen import codegen
from fiddle.experimental import serialization
from flax import traverse_util

from dsm import configs, console, datasets, envs, metrics, plotting, stade, train
from dsm.metric_writers import AimWriter, WanDBWriter
from dsm.state import State

_WORKDIR = epath.DEFINE_path("workdir", "logdir", "Working directory.")
_CHECKPOINT_FROM = flags.DEFINE_string(
    "checkpoint_from",
    None,
    "Checkpoint to load from, we'll only restore from this checkpoint if "
    "the checkpoint step is greater than the current step."
    "If not specified, will load from the latest checkpoint in the working directory.",
)
_PROFILE = flags.DEFINE_bool("profile", False, "Enable profiling.")
_AIM_FLAGS = ff.DEFINE_dict(
    "aim",
    repo=ff.String(None, "Repository directory."),
    experiment=ff.String("dsr", "Experiment name."),
    log_system_params=ff.Boolean(True, "Log system parameters."),
)
_WANDB_FLAGS = ff.DEFINE_dict(
    "wandb",
    save_code=ff.Boolean(False, "Save code."),
    tags=ff.StringList(None, "Tags."),
    name=ff.String(None, "Name."),
    group=ff.String(None, "Group."),
    mode=ff.Enum("online", ["online", "offline", "disabled"], "Mode."),
)
_METRIC_WRITER = flags.DEFINE_enum(
    "metric_writer",
    "wandb",
    ["aim", "wandb"],
    "Metric writer to use.",
)


jax.config.parse_flags_with_absl()


def _maybe_remove_absl_logger() -> None:
    if (absl_handler := absl_logging.get_absl_handler()) in logging.root.handlers:
        logging.root.removeHandler(absl_handler)


def _stop_progress_on_breakpoint(pbar: tqdm.tqdm) -> None:
    def _breakpointhook(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        frame = None
        if currentframe := inspect.currentframe():
            frame = currentframe.f_back
        pbar._prog.stop()  # pyright: ignore
        sys.__breakpointhook__(frame)
        pbar._prog.start()  # pyright: ignore

    sys.breakpointhook = _breakpointhook


def _write_experiment_artifacts(buildable: fdl.Buildable, workdir: epath.Path) -> None:
    if not (config_json_path := workdir / "config.json").exists():
        config_json_path.write_text(serialization.dump_json(buildable, indent=2))
    if not (config_py_path := workdir / "config.py").exists():
        config_py_path.write_text("\n".join(codegen.codegen_dot_syntax(buildable).lines()))


def _create_metric_writer(workdir: epath.Path) -> metric_writers.MetricWriter:
    run_hash_path = workdir / ".run-hash"
    run_hash = run_hash_path.read_text().strip() if run_hash_path.exists() else None

    match _METRIC_WRITER.value:
        case "aim":
            writer = AimWriter(**_AIM_FLAGS.value, run_hash=run_hash)
            run_hash = writer.run.hash
        case "wandb":
            writer = WanDBWriter(**_WANDB_FLAGS.value, id=run_hash, resume="must" if run_hash else "never")
            run_hash = writer.run.id
        case _:
            raise ValueError(f"Unknown metric writer: {_METRIC_WRITER.value}")

    if not run_hash_path.exists():
        run_hash_path.write_text(run_hash)

    return writer


def _maybe_restore_state(checkpoint_manager: orbax.checkpoint.CheckpointManager, state: State) -> State:
    latest_step = checkpoint_manager.latest_step()

    def _restore_state(step: int, directory: os.PathLike[str] | None = None) -> State:
        logging.info(f"Restoring checkpoint from {directory or checkpoint_manager.directory} at step {step}.")
        restored = checkpoint_manager.restore(
            step,
            {"generator": state.generator, "discriminator": state.discriminator},
            directory=directory,
        )
        [g_state, d_state] = operator.itemgetter("generator", "discriminator")(restored)
        return State(step=jnp.int32(step), generator=g_state, discriminator=d_state)

    if _CHECKPOINT_FROM.value and (checkpoint_steps := orbax.checkpoint.utils.checkpoint_steps(_CHECKPOINT_FROM.value)):
        logging.info(f"Found checkpoint directory {_CHECKPOINT_FROM.value} with steps {checkpoint_steps}.")
        latest_checkpoint_step = max(checkpoint_steps)
        if not latest_step or latest_checkpoint_step > latest_step:
            return _restore_state(latest_checkpoint_step, _CHECKPOINT_FROM.value)
    if latest_step:
        return _restore_state(latest_step)

    logging.info("No checkpoint found.")
    return state


def main(_) -> None:
    _maybe_remove_absl_logger()

    logging.info("JAX Default Backend: %s", jax.default_backend())
    logging.info("JAX Process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX Process Count: %d", jax.process_count())
    logging.info("JAX Devices: %r", jax.devices())
    logging.info("JAX Device Count: %d", jax.device_count())
    logging.info("JAX Local Devices: %r", jax.local_devices())
    logging.info("JAX Local Device Count: %d", jax.local_device_count())

    buildable = fdl_flags.create_buildable_from_flags(configs)

    logging.info(printing.as_str_flattened(buildable))
    config: configs.Config = fdl.build(buildable)
    workdir: epath.Path = _WORKDIR.value
    workdir.mkdir(parents=True, exist_ok=True)

    metric_writer = _create_metric_writer(workdir)
    # TODO(jfarebro): use printing.as_dict
    # metric_writer.write_hparams(_fiddle_config_to_hparams(buildable))
    _write_experiment_artifacts(buildable, workdir)

    pbar = tqdm.tqdm(total=config.num_grad_updates, options=dict(console=console), disable=not sys.stderr.isatty())
    _stop_progress_on_breakpoint(pbar)

    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        workdir,
        checkpointers={
            "generator": orbax.checkpoint.PyTreeCheckpointer(),
            "discriminator": orbax.checkpoint.PyTreeCheckpointer(),
        },
        options=orbax.checkpoint.CheckpointManagerOptions(create=True, max_to_keep=2),
    )

    def metrics_callback(infos: dict[str, jax.Array]) -> None:
        step = infos.pop("step").item()
        if not jax.tree_util.tree_all(
            jax.tree_util.tree_map(np.isfinite, infos),
        ):
            logging.error("Non-finite metrics (%r) at step %d.", infos, step)
        pbar.update(step - pbar.n)
        metric_writer.write_scalars(step, traverse_util.flatten_dict(infos, sep="/"))  # type: ignore

    def plot_callback(state: State) -> None:
        step = state.step.item()
        metric_writer.write_images(
            step,
            plotting.plot_samples(state.generator, jax.random.PRNGKey(0), config=config),
        )
        # TODO: comment out for one-step model
        histograms, statistics = metrics.compute_distribution_metrics(
            state.generator, jax.random.PRNGKey(0), config=config
        )
        metric_writer.write_scalars(step, statistics)
        metric_writer.write_histograms(step, histograms)
        metric_writer.write_images(
            step,
            plotting.plot_cdf(state.generator, jax.random.PRNGKey(0), config=config),
        )

    def checkpoint_callback(state: State) -> None:
        step = state.step.item()
        checkpoint_manager.save(
            step,
            {"generator": state.generator, "discriminator": state.discriminator},
            metrics={
                "generator": state.generator.metrics.compute(),
                "discriminator": state.generator.metrics.compute(),
            },
        )

    env = envs.make(config.env)
    env = stade.GymEnvWrapper(env, with_infos=False, seed=None)
    rng = np.random.default_rng(config.seed)

    data = datasets.make_dataset(config.env)

    rng_key = jax.random.PRNGKey(rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max))
    rng_key, state_rng_key = jax.random.split(rng_key)
    state = train.make_state(state_rng_key, typing.cast(specs.DiscreteArray, env.observation_spec()), config)
    state = _maybe_restore_state(checkpoint_manager, state)
    pbar.update(state.step.item())

    with contextlib.ExitStack() as stack:
        if _PROFILE.value:
            if config.num_grad_updates > 10_000:
                logging.warning("Profiling with a large number of steps (%s).", config.num_grad_updates)
            stack.enter_context(jax.profiler.trace(workdir.as_posix(), create_perfetto_link=True))
        state = train.train(
            state,
            data,
            rng=rng_key,
            metrics_callback=metrics_callback,
            plot_callback=plot_callback,
            checkpoint_callback=checkpoint_callback,
            config=config,
        )

    checkpoint_callback(state)
    logging.info("Done.")


if __name__ == "__main__":
    fiddle.extensions.jax.enable()

    logging.getLogger("jax").setLevel(logging.INFO)
    jax.config.update("jax_numpy_rank_promotion", "raise")

    app.run(main, flags_parser=fdl_flags.flags_parser)
