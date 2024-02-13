import logging
from typing import Any, Mapping

import aim
from clu import metric_writers
from clu.metric_writers.interface import Array

import wandb


class AimLogHandler(logging.Handler):
    def __init__(self, run: aim.Run) -> None:
        super().__init__()
        self.run = run

    def emit(self, record: logging.LogRecord) -> None:
        match record.levelno:
            case logging.DEBUG:
                self.run.log_debug(record.getMessage())
            case logging.INFO:
                self.run.log_info(record.getMessage())
            case logging.WARNING:
                self.run.log_warning(record.getMessage())
            case logging.ERROR | logging.CRITICAL:
                self.run.log_error(record.getMessage())
            case _:
                raise ValueError(f"Unknown log level: {record.levelno}")


class AimWriter(metric_writers.MetricWriter):
    def __init__(self, **kwargs):
        self.run = aim.Run(**kwargs)
        logging.getLogger("aim").addHandler(AimLogHandler(self.run))

    def write_summaries(self, step: int, values: Mapping[str, Array], metadata: Mapping[str, Any] | None = None):
        for key, value in values.items():
            self.run.track(value, name=key, step=step, context=metadata)  # type: ignore
        self.run.report_progress()

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self.run.track(aim.Image(value), name=key, step=step)
        self.run.report_progress()

    def write_scalars(self, step: int, scalars: Mapping[str, float]):
        for key, value in scalars.items():
            self.run.track(value, name=key, step=step)
        self.run.report_progress()

    def write_hparams(self, hparams: Mapping[str, Any]):
        self.run["hparams"] = hparams

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        raise NotImplementedError

    def write_histograms(self, step: int, arrays: Mapping[str, Array], num_buckets: Mapping[str, int] | None = None):
        for key, array in arrays.items():
            dist = aim.Distribution(
                array,
                bin_count=num_buckets[key] if num_buckets else 64,
            )
            self.run.track(dist, name=key, step=step)

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        raise NotImplementedError

    def write_texts(self, step: int, texts: Mapping[str, str]):
        raise NotImplementedError

    def close(self):
        self.run.close()

    def flush(self):
        pass


class WanDBWriter(metric_writers.MetricWriter):
    def __init__(self, **kwargs):
        self.run: wandb.wandb_sdk.wandb_run.Run = wandb.init(**kwargs)  # type: ignore

    def write_summaries(self, step: int, values: Mapping[str, Array], metadata: Mapping[str, Any] | None = None):
        for key, value in values.items():
            self.run.log({key: value, **(metadata or {})}, step=step)

    def write_images(self, step: int, images: Mapping[str, Array]):
        for key, value in images.items():
            self.run.log({key: [wandb.Image(value)]}, step=step)

    def write_scalars(self, step: int, scalars: Mapping[str, float]):
        for key, value in scalars.items():
            self.run.log({key: value}, step=step)

    def write_hparams(self, hparams: Mapping[str, Any]):
        self.run.config.update(hparams)

    def write_audios(self, step: int, audios: Mapping[str, Array], *, sample_rate: int):
        raise NotImplementedError

    def write_histograms(self, step: int, arrays: Mapping[str, Array], num_buckets: Mapping[str, int] | None = None):
        self.run.log(
            {
                key: wandb.Histogram(array, num_bins=num_buckets[key] if num_buckets else 64)  # type: ignore
                for key, array in arrays.items()
            },
            step=step,
        )

    def write_videos(self, step: int, videos: Mapping[str, Array]):
        raise NotImplementedError

    def write_texts(self, step: int, texts: Mapping[str, str]):
        raise NotImplementedError

    def close(self):
        pass

    def flush(self):
        pass
