# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Celo-compatible patched single-task training curves.

Adapted from `amoudgl/celo/celo/eval_training.py` at commit 66d9761. The
important protocol details are: eval every 10 steps by default, splits
train/outer_valid/test, final eval over last_eval_batches, and the metrics_every
zip bug fix from Celo's local copy.
"""

from __future__ import annotations

import functools
from typing import Any, Iterator, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import tqdm
from absl import logging
from jax import lax
from learned_optimization import jax_utils, profile, summary, tree_utils
from learned_optimization.optimizers import base as opt_base
from learned_optimization.tasks import base as tasks_base

OptState = Any
Data = Any
PRNGKey = jnp.ndarray


@functools.partial(jax.jit, static_argnames=("task", "opt", "pmap_axis_name", "with_metrics"))
def _next_state(
    task: tasks_base.Task,
    opt: opt_base.Optimizer,
    opt_state: OptState,
    data: Any,
    key: PRNGKey,
    pmap_axis_name: Optional[str] = None,
    is_valid: bool = False,
    with_metrics: bool = False,
) -> Tuple[OptState, jnp.ndarray, PRNGKey, Mapping[str, jnp.ndarray]]:
    def fn(opt_state, key, data):
        key, key1 = jax.random.split(key)
        p, s = opt.get_params_state(opt_state)
        (loss, state), grad = jax.value_and_grad(task.loss_with_state, has_aux=True)(p, s, key1, data)
        if pmap_axis_name:
            grad = lax.pmean(grad, pmap_axis_name)
            loss = lax.pmean(loss, pmap_axis_name)
        key, key1 = jax.random.split(key)
        next_opt_state = opt.update(opt_state, grad, loss=loss, model_state=state, is_valid=is_valid, key=key1)
        return next_opt_state, loss, key

    if with_metrics:
        key, summary_key = jax.random.split(key)
        (next_opt_state, loss, key), metrics = summary.with_summary_output_reduced(fn)(
            opt_state, key, data, sample_rng_key=summary_key
        )
        key, key1 = jax.random.split(key)
        metrics = summary.aggregate_metric_list([metrics], use_jnp=True, key=key1)
    else:
        next_opt_state, loss, key = fn(opt_state, key, data)
        metrics = {}
    return next_opt_state, loss, key, metrics


@functools.partial(jax.jit, static_argnames=("task", "opt", "pmap_axis_name"))
def _loss_and_aux(
    task: tasks_base.Task,
    opt: opt_base.Optimizer,
    opt_state: OptState,
    data: Data,
    key: PRNGKey,
    pmap_axis_name: Optional[str] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray]]:
    p, s = opt.get_params_state(opt_state)
    loss, _, aux = task.loss_with_state_and_aux(p, s, key, data)
    if pmap_axis_name:
        loss = lax.pmean(loss, pmap_axis_name)
        aux = lax.pmean(aux, pmap_axis_name)
    norm_fn = getattr(task, "normalizer", lambda x: x)
    return loss, norm_fn(loss), aux


def _batch_eval(
    task: tasks_base.Task,
    opt: opt_base.Optimizer,
    opt_state: Any,
    key: PRNGKey,
    data_iter: Iterator[Any],
    eval_batches: int,
    device: Optional[jax.Device] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray]]:
    eval_losses = []
    eval_norm_losses = []
    eval_auxs = []
    for _ in range(eval_batches):
        key, key1 = jax.random.split(key)
        batch = next(data_iter) if data_iter else ()
        if device:
            batch = jax.device_put(batch, device=device)
        loss, norm_loss, aux = _loss_and_aux(task, opt, opt_state, batch, key=key1)
        eval_losses.append(loss)
        eval_norm_losses.append(norm_loss)
        eval_auxs.append(aux)
    return (
        onp.mean(eval_losses),
        onp.mean(eval_norm_losses),
        jax.tree_util.tree_map(onp.mean, tree_utils.tree_zip_onp(eval_auxs)),
    )


@profile.wrap()
def single_task_training_curves(
    task: tasks_base.Task,
    opt: opt_base.Optimizer,
    num_steps: int,
    key: PRNGKey,
    eval_every: int = 10,
    eval_batches: int = 5,
    last_eval_batches: int = 20,
    eval_task: Optional[tasks_base.Task] = None,
    device: Optional[jax.Device] = None,
    metrics_every: Optional[int] = None,
    summary_writer: Optional[summary.SummaryWriterBase] = None,
) -> Mapping[str, jnp.ndarray]:
    if eval_task is None:
        eval_task = task

    splits = ["train", "outer_valid", "test"]
    with profile.Profile("setup"):
        key = jax.device_put(key, device)
        key, key1 = jax.random.split(key)
        params, state = jax_utils.cached_jit(task.init_with_state)(key1)
        opt_state = jax_utils.cached_jit(opt.init, static_argnames=("num_steps",))(
            params, model_state=state, num_steps=num_steps
        )

    losses = []
    eval_auxs = []
    use_data = task.datasets is not None
    train_xs = []
    eval_xs = []
    metrics = []
    metrics_xs = []

    for i in tqdm.trange(num_steps + 1, position=0):
        with profile.Profile("eval"):
            m = {}
            if i % eval_every == 0 and eval_batches:
                on_last = i == num_steps
                for split in splits:
                    key, key1 = jax.random.split(key)
                    loss, loss_normalized, aux = _batch_eval(
                        eval_task,
                        opt,
                        opt_state,
                        key1,
                        task.datasets.split(split) if use_data else (),
                        eval_batches if not on_last else last_eval_batches,
                        device=device,
                    )
                    m[f"eval/{split}/loss"] = loss
                    m[f"eval/{split}/loss_normalized"] = loss_normalized
                    for aux_key, value in aux.items():
                        m[f"eval/{split}/{aux_key}"] = value
                eval_auxs.append(m)
                if summary_writer:
                    for metric_key, value in m.items():
                        summary_writer.scalar(metric_key, value, step=i)
                eval_xs.append(i)

        with profile.Profile("get_batch"):
            batch = next(task.datasets.train) if use_data else ()
        with profile.Profile("put_batch_and_split"):
            batch = jax.device_put(batch, device=device)
        with profile.Profile("next_state"):
            with_metrics = False if metrics_every is None else i % metrics_every == 0
            opt_state, loss, key, m = _next_state(task, opt, opt_state, batch, key, with_metrics=with_metrics)
            losses.append(loss)
            train_xs.append(i)
            if summary_writer:
                summary_writer.scalar("train/loss", loss, step=i)
            if metrics_every and i % metrics_every == 0:
                if summary_writer:
                    for metric_key, value in m.items():
                        agg, clean_key = metric_key.split("||")
                        if agg in ["mean", "sample"]:
                            summary_writer.scalar(clean_key, value, step=i)
                        elif agg == "tensor":
                            summary_writer.tensor(clean_key, value, step=i)
                        else:
                            logging.warning("Dropping unsupported aggregation type %s for key %s.", agg, clean_key)
                metrics.append(m)
                metrics_xs.append(i)

    ret = {
        "train/xs": onp.asarray(train_xs),
        "train/loss": onp.asarray(losses),
    }
    if metrics_every:
        stacked_metrics = tree_utils.tree_zip_onp(metrics)
        ret = {**ret, **{f"train/metrics/{k}": v for k, v in stacked_metrics.items()}}
        ret["train/metrics/xs"] = onp.asarray(metrics_xs)
    if eval_batches:
        stacked_eval = tree_utils.tree_zip_onp(eval_auxs)
        ret["eval/xs"] = onp.asarray(eval_xs)
        ret["eval/last_eval_batches"] = onp.asarray(last_eval_batches)
        ret["eval/eval_batches"] = onp.asarray(eval_batches)
        ret = {**ret, **stacked_eval}
    return ret

