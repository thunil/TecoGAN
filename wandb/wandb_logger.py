from tqdm import tqdm

import cv2
import numpy as np

from torch import Tensor

import logging
import os
import pkg_resources as pkg
import sys
from pathlib import Path
from typing import Union

__all__ = ["WandBLogger"]

WANDB_ARTIFACT_PREFIX = "wandb-artifact://"

class WandBLogger:
    """
    The `WandBLogger` provides an easy integration with
    Weights & Biases logging. Each monitored metric is automatically
    logged to a dedicated Weights & Biases project dashboard.

    .. note::
    The wandb log files are placed by default in "./wandb/" unless specified.
    """

    def __init__(
        self,
        project_name: str = "TecoGAN",
        run_name: str = None,
        save_code: bool = True,
        config: object = None,
        dir: Union[str, Path] = None,
        model: object = None,
        job_type: str = "training",
        rank: int = 0,
        params: dict = None,
    ) -> None:
        """
        Creates an instance of the `WandBLogger`.
        :param project_name: Name of the W&B project.
        :param run_name: Name of the W&B run.
        :param save_code: Saves the main training script to W&B.
        :param dir: Path to the local log directory for W&B logs to be saved at.
        :param model: Model checkpoint to be logged to W&B.
        :param config: Syncs hyper-parameters and config values used to W&B.
        :rank: The rank of the current process.
        :param params: All arguments for wandb.init() function call.
        Visit https://docs.wandb.ai/ref/python/init to learn about all
        wand.init() parameters.
        """

        self.project_name = project_name
        self.run_name = run_name
        self.save_code = save_code
        self.dir = dir
        self.config = config
        self.model = model
        self.job_type = job_type
        self.rank = rank
        self.params = params

        self._import_wandb()
        self._args_parse()
        self._before_job()

        self.current_epoch = 0

    def _import_wandb(self):
        """Imports Weights & Biases package

        Raises:
            ImportError: If the Weights & Biases package is not installed.
        """
        try:
            import wandb

            assert hasattr(wandb, "__version__")
            if pkg.parse_version(wandb.__version__) >= pkg.parse_version(
                "0.12.2"
            ) and self.rank in [0, -1]:
                wandb.login(timeout=30)
            else:
                logging.warning(
                    "wandb latest version is higher than 0.12.2, please update to 0.12.2"
                )
        except (ImportError, AssertionError):
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def _args_parse(self):
        """Parses the arguments for wandb.init() function call."""
        self.init_kwargs = {
            "project": self.project_name,
            "name": self.run_name,
            "save_code": self.save_code,
            "dir": self.dir,
            "job_type": self.job_type,
            "config": vars(self.config) if self.config else None,
        }
        if self.params:
            self.init_kwargs.update(self.params)

    def _before_job(self):
        """Initializes the Weights & Biases logger."""
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()
        if self.model is not None:
            self.wandb.watch(self.model)
        self.wandb.run._label(repo="TecoGAN")

    def log_metrics(
        self,
        key: str = None,
        value: Union[int, float, Tensor] = None,
        step: Union[int, float] = None,
    ) -> None:
        """Logs metrics to Weights & Biases dashboard.

        Args:
            key: Name of the metric.
            value: Value of the metric.
        """
        self.wandb.log({key: value}, step=step)

    def log_image(
        self,
        key: str = None,
        image: Union[np.ndarray, Tensor] = None,
        step: Union[int, float] = None,
    ) -> None:
        """Logs images to Weights & Biases dashboard.

        Args:
            key: Name of the image.
            image: Image to be logged.
        """
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if isinstance(image, Tensor):
            image = image.numpy()
        self.wandb.log({key: self.wandb.Image(image)}, step=step)

    def log_artifact(
        self,
        key: str = None,
        path: Union[str, Path] = None,
        step: Union[int, float] = None,
    ) -> None:
        """Logs artifacts to Weights & Biases dashboard.

        Args:
            key: Name of the artifact.
            path: Path to the artifact.
        """
        self.wandb.log({key: self.wandb.Artifact(WANDB_ARTIFACT_PREFIX + path)}, step=step)  

    def log_table(self):
        """Logs table to Weights & Biases dashboard."""
        pass      
