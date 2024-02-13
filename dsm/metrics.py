import logging
from typing import NamedTuple

import jax
import numpy as np
import numpy.typing as npt
from scipy import stats

from dsm import datasets, plotting, rewards
from dsm.configs import Config
from dsm.plotting import utils as plot_utils
from dsm.state import FittedValueTrainState


class MetricResult(NamedTuple):
    statistic: float
    pvalue: float | None


def compute_distribution_metrics(
    state: FittedValueTrainState,
    rng: jax.random.KeyArray,
    *,
    config: Config,
) -> tuple[dict[str, npt.NDArray], dict[str, float]]:
    policy = datasets.make_policy(config.env)

    metrics = []
    for reward_fn_name, reward_fn in getattr(rewards, config.env).items():
        logging.info(f"Computing distribution metrics for {config.env} reward function {reward_fn_name}")
        for _, source in zip(*plotting.source_states(config.env)):
            dsr_return_distribution = plot_utils.return_distribution(
                state,
                rng,
                source,
                policy=policy,
                reward_fn=reward_fn,
                num_samples=config.plot_num_samples,
                config=config,
            )

            mc_returns = plot_utils.compute_mc_returns(
                config.env,
                source,
                reward_fn=reward_fn,
                with_truncation=False,
                config=config,
            )

            cvm = stats.cramervonmises_2samp(dsr_return_distribution, mc_returns)
            wasserstein = stats.wasserstein_distance(dsr_return_distribution, mc_returns)
            ks = stats.ks_2samp(dsr_return_distribution, mc_returns)

            metrics.append({
                "cvm": MetricResult(cvm.statistic, cvm.pvalue),  # type: ignore
                "wasserstein": MetricResult(wasserstein, None),
                "ks": MetricResult(ks.statistic, ks.pvalue),  # type: ignore
            })

    all_stats = jax.tree_util.tree_map(
        lambda *args: list(filter(None, map(lambda x: x.statistic, args))),
        *metrics,
        is_leaf=lambda x: isinstance(x, MetricResult),
    )
    all_pvalues = jax.tree_util.tree_map(
        lambda *args: list(filter(None, map(lambda x: x.pvalue, args))),
        *metrics,
        is_leaf=lambda x: isinstance(x, MetricResult),
    )

    def _compute_statistics(statistics: dict[str, list[float]], *, prefix: str = "") -> dict[str, float]:
        return {  # type: ignore
            **{f"{prefix}{key}/mean": np.mean(samples) for key, samples in statistics.items() if samples},
            **{f"{prefix}{key}/min": np.min(samples) for key, samples in statistics.items() if samples},
            **{f"{prefix}{key}/max": np.max(samples) for key, samples in statistics.items() if samples},
            **{f"{prefix}{key}/median": np.median(samples) for key, samples in statistics.items() if samples},
        }

    statistics = _compute_statistics(all_stats, prefix="stat/") | _compute_statistics(all_pvalues, prefix="pvalue/")
    histograms = {
        **{f"stat/{key}": np.array(samples) for key, samples in all_stats.items() if samples},
        **{f"pvalue/{key}": np.array(samples) for key, samples in all_pvalues.items() if samples},
    }

    return histograms, statistics
