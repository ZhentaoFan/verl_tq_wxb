# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket
import uuid
import warnings
import math
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo import core_algos
from verl.trainer.ppo import ray_trainer as base_ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type
from verl.utils.tracking import Tracking


def compute_advantage_for_multi_trajectories(
    data: DataProto,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> DataProto:
    if (
        adv_estimator != core_algos.AdvantageEstimator.GRPO
        or "session_id" not in data.non_tensor_batch
        or "trajectory_index" not in data.non_tensor_batch
    ):
        return base_ray_trainer.compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    uids = data.non_tensor_batch["uid"]
    session_ids = data.non_tensor_batch["session_id"]
    trajectory_indices = data.non_tensor_batch["trajectory_index"]

    final_session_rows: dict[tuple[str, int], tuple[int, int]] = {}
    row_session_keys = []
    for row_index, (uid, session_id, trajectory_index) in enumerate(
        zip(uids, session_ids, trajectory_indices, strict=True)
    ):
        session_key = (str(uid), int(session_id))
        row_session_keys.append(session_key)
        trajectory_index = int(trajectory_index)
        if session_key not in final_session_rows or final_session_rows[session_key][0] < trajectory_index:
            final_session_rows[session_key] = (trajectory_index, row_index)

    final_indices = [row_index for _, row_index in final_session_rows.values()]
    final_data = base_ray_trainer.compute_advantage(
        data.select_idxs(final_indices),
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )

    final_advantages = final_data.batch["advantages"]
    final_response_mask = final_data.batch["response_mask"]
    first_valid_token = final_response_mask.to(torch.int64).argmax(dim=1)
    final_session_scores = final_advantages[
        torch.arange(final_advantages.size(0), device=final_advantages.device), first_valid_token
    ]

    session_key_to_local_index = {
        session_key: local_index for local_index, session_key in enumerate(final_session_rows.keys())
    }
    row_to_local_index = torch.tensor(
        [session_key_to_local_index[session_key] for session_key in row_session_keys],
        dtype=torch.long,
        device=final_session_scores.device,
    )
    broadcast_scores = final_session_scores[row_to_local_index].unsqueeze(-1) * data.batch["response_mask"]
    data.batch["advantages"] = broadcast_scores
    data.batch["returns"] = broadcast_scores
    return data


class MultiTrajectoryRayPPOTrainer(RayPPOTrainer):
    @staticmethod
    def _to_bool_array(values) -> np.ndarray:
        return np.array([bool(value) for value in values], dtype=bool)

    @staticmethod
    def _to_int_array(values) -> np.ndarray:
        return np.array([int(value) for value in values], dtype=np.int64)

    def _get_balance_padding_multiple(self) -> int:
        multiples = [self.actor_rollout_wg.world_size]

        actor_mini_batch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size", None)
        if actor_mini_batch_size:
            multiples.append(int(actor_mini_batch_size) * int(self.config.actor_rollout_ref.rollout.n))

        if self.use_critic:
            critic_mini_batch_size = self.config.critic.get("ppo_mini_batch_size", None)
            if critic_mini_batch_size:
                multiples.append(int(critic_mini_batch_size) * int(self.config.actor_rollout_ref.rollout.n))

        return math.lcm(*[multiple for multiple in multiples if multiple > 0])

    def _pad_batch_for_training(
        self,
        batch: DataProto,
        target_multiple: int,
        metrics: dict,
        logging_prefix: str,
    ) -> DataProto:
        original_batch_size = len(batch)
        metrics[f"{logging_prefix}/padding_target_multiple"] = target_multiple
        metrics[f"{logging_prefix}/padding_size"] = 0
        metrics[f"{logging_prefix}/padded_batch_size"] = original_batch_size

        if original_batch_size == 0:
            return batch

        padding_size = (-original_batch_size) % target_multiple
        metrics[f"{logging_prefix}/padding_size"] = padding_size
        metrics[f"{logging_prefix}/padded_batch_size"] = original_batch_size + padding_size
        if padding_size == 0:
            return batch

        padding_indices = [(original_batch_size - 1 - (i % original_batch_size)) for i in range(padding_size)]
        padding_indices.reverse()
        padding_batch = batch.select_idxs(np.array(padding_indices, dtype=np.int64))
        padded_batch = DataProto.concat([batch, padding_batch])
        padded_batch.meta_info = deepcopy(batch.meta_info)
        return padded_batch

    def _prepare_train_batch(
        self,
        batch: DataProto,
        metrics: dict,
        logging_prefix: str = "global_seqlen",
        keep_minibatch: bool = False,
    ) -> DataProto:
        train_batch = batch.select_idxs(np.arange(len(batch), dtype=np.int64))
        train_batch.meta_info = deepcopy(batch.meta_info)
        target_multiple = self._get_balance_padding_multiple()
        train_batch = self._pad_batch_for_training(train_batch, target_multiple, metrics, logging_prefix)
        if self.config.trainer.balance_batch:
            super()._balance_batch(train_batch, metrics, logging_prefix, keep_minibatch)
        return train_batch

    def _run_worker_op_with_padding(
        self,
        batch: DataProto,
        worker_group,
        worker_fn,
        metrics: Optional[dict] = None,
        logging_prefix: Optional[str] = None,
    ) -> DataProto:
        padded_batch, pad_size = pad_dataproto_to_divisor(batch, worker_group.world_size)
        if metrics is not None and logging_prefix is not None:
            metrics[f"{logging_prefix}/padding_size"] = pad_size
            metrics[f"{logging_prefix}/padded_batch_size"] = len(padded_batch)
        result = worker_fn(padded_batch)
        if pad_size > 0:
            result = unpad_dataproto(result, pad_size=pad_size)
        return result

    def _expand_batch_for_rollout_outputs(self, batch: DataProto, rollout_output: DataProto) -> DataProto:
        if "source_row_index" not in rollout_output.non_tensor_batch:
            return batch
        return batch.select_idxs(self._to_int_array(rollout_output.non_tensor_batch["source_row_index"]))

    def _select_final_rollout_outputs(self, rollout_output: DataProto) -> DataProto:
        if "is_session_final" not in rollout_output.non_tensor_batch:
            return rollout_output
        final_mask = self._to_bool_array(rollout_output.non_tensor_batch["is_session_final"])
        if final_mask.all():
            return rollout_output
        return rollout_output.select_idxs(final_mask)

    def _validate(self):
        if not self.async_rollout_mode:
            return super()._validate()

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        dump_all_trajectories = bool(val_data_dir) and bool(
            self.config.trainer.get("validation_dump_all_trajectories", False)
        )
        dump_inputs = []
        dump_outputs = []
        dump_gts = []
        dump_scores = []
        dump_extra_infos_dict: dict[str, list] = defaultdict(list)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            eval_source_batch = test_batch.select_idxs(np.arange(len(test_batch), dtype=np.int64))

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            original_test_gen_batch_size = len(test_gen_batch)
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            if "source_row_index" in test_output_gen_batch_padded.non_tensor_batch:
                valid_mask = self._to_int_array(
                    test_output_gen_batch_padded.non_tensor_batch["source_row_index"]
                ) < original_test_gen_batch_size
                test_output_gen_batch = test_output_gen_batch_padded.select_idxs(valid_mask)
            else:
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            all_test_output_gen_batch = test_output_gen_batch
            test_output_gen_batch = self._select_final_rollout_outputs(test_output_gen_batch)
            prompt_eval_batch = self._expand_batch_for_rollout_outputs(eval_source_batch, test_output_gen_batch)
            eval_batch = self._expand_batch_for_rollout_outputs(test_batch, test_output_gen_batch)

            input_ids = prompt_eval_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(prompt_eval_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in prompt_eval_batch
            ]
            sample_gts.extend(ground_truths)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            eval_batch = eval_batch.union(test_output_gen_batch)
            eval_batch.meta_info["validate"] = True

            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(eval_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            if dump_all_trajectories:
                dump_prompt_eval_batch = self._expand_batch_for_rollout_outputs(eval_source_batch, all_test_output_gen_batch)
                dump_input_ids = dump_prompt_eval_batch.batch["input_ids"]
                dump_inputs.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in dump_input_ids])
                dump_outputs.extend(
                    [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in all_test_output_gen_batch.batch["responses"]]
                )
                dump_gts.extend(
                    [
                        item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                        for item in dump_prompt_eval_batch
                    ]
                )

                if "session_id" in all_test_output_gen_batch.non_tensor_batch:
                    final_session_scores = {
                        (str(uid), int(session_id)): score
                        for uid, session_id, score in zip(
                            prompt_eval_batch.non_tensor_batch["uid"],
                            self._to_int_array(test_output_gen_batch.non_tensor_batch["session_id"]),
                            scores,
                            strict=True,
                        )
                    }
                    trajectory_scores = [
                        final_session_scores.get((str(uid), int(session_id)))
                        for uid, session_id in zip(
                            dump_prompt_eval_batch.non_tensor_batch["uid"],
                            self._to_int_array(all_test_output_gen_batch.non_tensor_batch["session_id"]),
                            strict=True,
                        )
                    ]
                elif len(all_test_output_gen_batch) == len(scores):
                    trajectory_scores = list(scores)
                else:
                    trajectory_scores = [None] * len(all_test_output_gen_batch)

                dump_scores.extend(trajectory_scores)
                dump_extra_infos_dict["reward"].extend(trajectory_scores)
                dump_extra_infos_dict["uid"].extend([str(uid) for uid in dump_prompt_eval_batch.non_tensor_batch["uid"]])
                for key in [
                    "__num_turns__",
                    "history_trajectory_count",
                    "history_trajectory_index",
                    "is_compression_snapshot",
                    "is_session_final",
                    "session_id",
                    "source_row_index",
                    "tool_call_count",
                    "trajectory_index",
                ]:
                    if key in all_test_output_gen_batch.non_tensor_batch:
                        dump_extra_infos_dict[key].extend(all_test_output_gen_batch.non_tensor_batch[key].tolist())

            if "__num_turns__" in eval_batch.non_tensor_batch:
                sample_turns.append(eval_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(eval_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        if val_data_dir:
            dump_inputs_to_write = dump_inputs if dump_all_trajectories else sample_inputs
            dump_outputs_to_write = dump_outputs if dump_all_trajectories else sample_outputs
            dump_gts_to_write = dump_gts if dump_all_trajectories else sample_gts
            dump_scores_to_write = dump_scores if dump_all_trajectories else sample_scores
            dump_reward_extra_infos = dump_extra_infos_dict if dump_all_trajectories else reward_extra_infos_dict
            self._dump_generations(
                inputs=dump_inputs_to_write,
                outputs=dump_outputs_to_write,
                gts=dump_gts_to_write,
                scores=dump_scores_to_write,
                reward_extra_infos_dict=dump_reward_extra_infos,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = base_ray_trainer.process_validation_metrics(
            data_sources, sample_uids, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys())
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def fit(self):
        if not self.async_rollout_mode:
            return super().fit()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = base_ray_trainer.RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with base_ray_trainer.marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with base_ray_trainer.marked_timer("step", timing_raw):
                    with base_ray_trainer.marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == base_ray_trainer.AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with base_ray_trainer.marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            gen_baseline_output = self._select_final_rollout_outputs(gen_baseline_output)
                            baseline_batch = self._expand_batch_for_rollout_outputs(batch, gen_baseline_output)
                            batch = baseline_batch.union(gen_baseline_output)

                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self._run_worker_op_with_padding(
                                    batch,
                                    self.rm_wg,
                                    self.rm_wg.compute_rm_score,
                                    metrics=metrics,
                                    logging_prefix="dispatch_padding/rm_score_baseline",
                                )
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = base_ray_trainer.compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output, baseline_batch

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = self._expand_batch_for_rollout_outputs(batch, gen_batch_output)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = base_ray_trainer.compute_response_mask(batch)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with base_ray_trainer.marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self._run_worker_op_with_padding(
                                batch,
                                self.rm_wg,
                                self.rm_wg.compute_rm_score,
                                metrics=metrics,
                                logging_prefix="dispatch_padding/rm_score",
                            )
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = base_ray_trainer.compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = base_ray_trainer.compute_reward(batch, self.reward_fn)

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with base_ray_trainer.marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self._run_worker_op_with_padding(
                                batch,
                                self.actor_rollout_wg,
                                self.actor_rollout_wg.compute_log_prob,
                                metrics=metrics,
                                logging_prefix="dispatch_padding/old_log_prob",
                            )
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = base_ray_trainer.agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            metrics.update({"actor/entropy": entropy_agg.detach().item()})
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with base_ray_trainer.marked_timer(str(base_ray_trainer.Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self._run_worker_op_with_padding(
                                    batch,
                                    self.ref_policy_wg,
                                    self.ref_policy_wg.compute_ref_log_prob,
                                    metrics=metrics,
                                    logging_prefix="dispatch_padding/ref_log_prob",
                                )
                            else:
                                ref_log_prob = self._run_worker_op_with_padding(
                                    batch,
                                    self.actor_rollout_wg,
                                    self.actor_rollout_wg.compute_ref_log_prob,
                                    metrics=metrics,
                                    logging_prefix="dispatch_padding/ref_log_prob",
                                )
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with base_ray_trainer.marked_timer("values", timing_raw, color="cyan"):
                            values = self._run_worker_op_with_padding(
                                batch,
                                self.critic_wg,
                                self.critic_wg.compute_values,
                                metrics=metrics,
                                logging_prefix="dispatch_padding/values",
                            )
                            batch = batch.union(values)

                    with base_ray_trainer.marked_timer("adv", timing_raw, color="brown"):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = base_ray_trainer.apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        batch = compute_advantage_for_multi_trajectories(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                            config=self.config.algorithm,
                        )

                    train_batch = None
                    need_train_batch = self.use_critic or self.config.trainer.critic_warmup <= self.global_steps
                    if need_train_batch:
                        train_batch = self._prepare_train_batch(batch, metrics=metrics)

                    if self.use_critic:
                        with base_ray_trainer.marked_timer("update_critic", timing_raw, color="pink"):
                            assert train_batch is not None
                            critic_output = self.critic_wg.update_critic(train_batch)
                        metrics.update(base_ray_trainer.reduce_metrics(critic_output.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with base_ray_trainer.marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            assert train_batch is not None
                            train_batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            train_batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(train_batch)
                        metrics.update(base_ray_trainer.reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with base_ray_trainer.marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = base_ray_trainer.should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with base_ray_trainer.marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with base_ray_trainer.marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(base_ray_trainer.compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(base_ray_trainer.compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(base_ray_trainer.compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, base_ray_trainer.AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # use new model engine implementation
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
            # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        if config.actor_rollout_ref.rollout.mode == "sync":
            warnings.warn("spmd rollout mode is deprecated and will be removed in v0.6.2", stacklevel=2)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        # Ref policy has been fused into ActorRolloutRefWorker in new model engine,
        # we don't need to add a separate ref policy worker goup.
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = MultiTrajectoryRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
