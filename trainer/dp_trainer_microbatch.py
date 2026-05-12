from copy import deepcopy
import math
import os
import re
import json

import torch
import numpy as np

from config.configurator import configs
from models.bulid_model import build_model
from trainer.trainer import Trainer
from trainer.utils import log_exceptions


class DPMicrobatchTrainer(Trainer):
    DEFAULT_DP_TARGET_NAMES = {
        "user_embeds",
        "item_embeds",
        "user_embedding.weight",
        "item_embedding.weight",
    }

    def __init__(self, data_handler, logger):
        super().__init__(data_handler, logger)
        privacy_config = configs.get("privacy", {})
        train_size = max(len(self.data_handler.train_dataloader.dataset), 1)

        self.privacy_config = {
            "enable": privacy_config.get("enable", True),
            "microbatch_size": int(privacy_config.get("microbatch_size", 32)),
            "clip_norm": float(privacy_config.get("clip_norm", 1.0)),
            "noise_multiplier": float(privacy_config.get("noise_multiplier", 1.0)),
            "delta": float(privacy_config.get("delta", 1.0 / train_size)),
            "target_scope": str(privacy_config.get("target_scope", "all")).lower(),
            "target_names": privacy_config.get("target_names"),
            "save_suffix": privacy_config.get("save_suffix", "dpmb"),
            "disable_model_randomness": bool(privacy_config.get("disable_model_randomness", True)),
        }
        self.dp_step_count = 0

    def _named_trainable_parameters(self, model):
        return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]

    def _split_parameters(self, model):
        target_scope = self.privacy_config["target_scope"]
        target_names = self.privacy_config["target_names"]
        target_name_set = set(target_names) if isinstance(target_names, list) else self.DEFAULT_DP_TARGET_NAMES

        dp_parameters = []
        normal_parameters = []
        for name, parameter in self._named_trainable_parameters(model):
            if target_scope == "all":
                dp_parameters.append((name, parameter))
            elif target_scope == "selective" and name in target_name_set:
                dp_parameters.append((name, parameter))
            elif target_scope == "regex" and target_names and any(re.search(pattern, name) for pattern in target_names):
                dp_parameters.append((name, parameter))
            else:
                normal_parameters.append((name, parameter))
        return dp_parameters, normal_parameters

    def _zero_model_randomness(self, model):
        if not self.privacy_config["disable_model_randomness"]:
            return
        if hasattr(model, "keep_rate"):
            model.keep_rate = 1.0
        if hasattr(model, "mask_ratio"):
            model.mask_ratio = 0.0
        if hasattr(model, "edge_bias"):
            model.edge_bias = 0.0
        if hasattr(model, "is_training"):
            model.is_training = True

    def _split_microbatches(self, batch_data):
        microbatch_size = max(1, self.privacy_config["microbatch_size"])
        batch_size = len(batch_data[0])
        for start in range(0, batch_size, microbatch_size):
            end = min(start + microbatch_size, batch_size)
            yield [tensor[start:end] for tensor in batch_data], end - start

    def _create_grad_buffers(self, named_parameters):
        return {name: torch.zeros_like(parameter) for name, parameter in named_parameters}

    def _clone_gradients(self, named_parameters):
        gradient_map = {}
        for name, parameter in named_parameters:
            gradient_map[name] = None if parameter.grad is None else parameter.grad.detach().clone()
        return gradient_map

    def _set_requires_grad(self, named_parameters, requires_grad):
        original_states = {}
        for name, parameter in named_parameters:
            original_states[name] = parameter.requires_grad
            parameter.requires_grad_(requires_grad)
        return original_states

    def _restore_requires_grad(self, named_parameters, original_states):
        for name, parameter in named_parameters:
            parameter.requires_grad_(original_states[name])

    def _accumulate_clipped_grads(self, dp_parameters, grad_buffers):
        grad_norm_sq = torch.zeros(1, device=configs["device"])
        for _, parameter in dp_parameters:
            if parameter.grad is None:
                continue
            grad_norm_sq = grad_norm_sq + parameter.grad.detach().pow(2).sum()

        grad_norm = torch.sqrt(grad_norm_sq).item()
        clip_norm = self.privacy_config["clip_norm"]
        clip_coef = min(1.0, clip_norm / (grad_norm + 1e-12))

        for name, parameter in dp_parameters:
            if parameter.grad is None:
                continue
            grad_buffers[name].add_(parameter.grad.detach() * clip_coef)

        return grad_norm, clip_coef

    def _apply_noisy_gradients(self, dp_parameters, normal_parameters, dp_grad_buffers, normal_gradients, normalizer):
        clip_norm = self.privacy_config["clip_norm"]
        noise_multiplier = self.privacy_config["noise_multiplier"]

        for name, parameter in dp_parameters:
            accumulated_grad = dp_grad_buffers[name]
            noisy_grad = accumulated_grad / max(normalizer, 1.0)
            if noise_multiplier > 0:
                noise_std = noise_multiplier * clip_norm / max(normalizer, 1.0)
                noisy_grad = noisy_grad + torch.randn_like(noisy_grad) * noise_std
            parameter.grad = noisy_grad

        for name, parameter in normal_parameters:
            parameter.grad = normal_gradients[name]

    def _approximate_epsilon(self):
        sigma = self.privacy_config["noise_multiplier"]
        delta = self.privacy_config["delta"]
        steps = max(self.dp_step_count, 1)

        if sigma <= 0:
            return float("inf")

        epsilon_per_step = math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
        return math.sqrt(2.0 * steps * math.log(1.0 / delta)) * epsilon_per_step + steps * epsilon_per_step * (
            math.exp(epsilon_per_step) - 1.0
        )

    def train_epoch(self, model, epoch_idx):
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        loss_log_dict = {}
        model.train()
        self._zero_model_randomness(model)
        dp_parameters, normal_parameters = self._split_parameters(model)

        for _, tem in enumerate(train_dataloader):
            batch_data = [tensor.long().to(configs["device"]) for tensor in tem]
            dp_grad_buffers = self._create_grad_buffers(dp_parameters)
            batch_loss_dict = {}
            microbatch_count = 0
            sample_count = 0

            normal_gradients = {}
            if normal_parameters:
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss, batch_loss_dict = model.cal_loss(batch_data)
                batch_loss.backward()
                normal_gradients = self._clone_gradients(normal_parameters)

            self.optimizer.zero_grad(set_to_none=True)
            normal_grad_states = self._set_requires_grad(normal_parameters, False) if normal_parameters else {}

            for microbatch, microbatch_sample_count in self._split_microbatches(batch_data):
                self.optimizer.zero_grad(set_to_none=True)
                loss, loss_dict = model.cal_loss(microbatch)
                loss.backward()

                self._accumulate_clipped_grads(dp_parameters, dp_grad_buffers)
                microbatch_count += 1
                sample_count += microbatch_sample_count

                if not batch_loss_dict:
                    for loss_name, loss_value in loss_dict.items():
                        batch_loss_dict[loss_name] = 0.0
                for loss_name, loss_value in loss_dict.items():
                    batch_loss_dict[loss_name] = batch_loss_dict.get(loss_name, 0.0) + float(loss_value)

            if normal_parameters:
                self._restore_requires_grad(normal_parameters, normal_grad_states)

            self.optimizer.zero_grad(set_to_none=True)
            normalizer = float(microbatch_count)
            self._apply_noisy_gradients(
                dp_parameters,
                normal_parameters,
                dp_grad_buffers,
                normal_gradients,
                normalizer,
            )
            self.optimizer.step()
            self.dp_step_count += 1

            for loss_name, total_loss_value in batch_loss_dict.items():
                averaged_loss = total_loss_value / max(microbatch_count, 1) / len(train_dataloader)
                loss_log_dict[loss_name] = loss_log_dict.get(loss_name, 0.0) + averaged_loss

        if "log_loss" in configs["train"] and configs["train"]["log_loss"]:
            self.logger.log(loss_log_dict, save_to_log=False, print_to_console=True)

    @log_exceptions
    def train(self, model):
        now_patience = 0
        best_epoch = 0
        best_recall = -1e9
        best_state_dict = deepcopy(model.state_dict())
        self.create_optimizer(model)
        train_config = configs["train"]
        dp_parameters, normal_parameters = self._split_parameters(model)

        self.logger.log(
            "Enable microbatch DP training: microbatch_size={}, clip_norm={}, noise_multiplier={}, delta={}, target_scope={}, dp_params={}, normal_param_count={}, disable_model_randomness={}.".format(
                self.privacy_config["microbatch_size"],
                self.privacy_config["clip_norm"],
                self.privacy_config["noise_multiplier"],
                self.privacy_config["delta"],
                self.privacy_config["target_scope"],
                [name for name, _ in dp_parameters],
                len(normal_parameters),
                self.privacy_config["disable_model_randomness"],
            )
        )

        for epoch_idx in range(train_config["epoch"]):
            self.train_epoch(model, epoch_idx)
            approx_epsilon = self._approximate_epsilon()
            self.logger.log(
                "Epoch {} DP progress: approx_epsilon={:.4f}, delta={}, steps={}.".format(
                    epoch_idx,
                    approx_epsilon,
                    self.privacy_config["delta"],
                    self.dp_step_count,
                )
            )
            if epoch_idx % train_config["test_step"] == 0:
                eval_result = self.evaluate(model, epoch_idx)

                if eval_result["recall"][-1] > best_recall:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_recall = eval_result["recall"][-1]
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    now_patience += 1

                if now_patience == configs["train"]["patience"]:
                    break

        model = build_model(self.data_handler).to(configs["device"])
        model.load_state_dict(best_state_dict)
        self.evaluate(model)

        model = build_model(self.data_handler).to(configs["device"])
        model.load_state_dict(best_state_dict)
        test_result = self.test(model)

        self.save_model(model)
        self.logger.log(
            "Best Epoch {}. Final test result: {}. DP config: {}. Approx privacy summary: (approx_epsilon={:.4f}, delta={}, steps={}).".format(
                best_epoch,
                test_result,
                self.privacy_config,
                self._approximate_epsilon(),
                self.privacy_config["delta"],
                self.dp_step_count,
            )
        )

    def save_model(self, model):
        if configs["train"]["save_model"]:
            model_state_dict = model.state_dict()
            model_name = configs["model"]["name"]
            dataset_name = configs["data"]["name"]
            seed = configs["train"]["seed"]
            suffix = self.privacy_config.get("save_suffix", "dpmb")
            run_id = configs.get("run", {}).get("id")

            def make_json_safe(obj):
                if isinstance(obj, dict):
                    return {key: make_json_safe(value) for key, value in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [make_json_safe(value) for value in obj]
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.generic):
                    return obj.item()
                return obj

            if not configs["tune"]["enable"]:
                save_dir_path = "./checkpoint/{}".format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)

                file_stem = "{}-{}-{}-{}".format(model_name, dataset_name, seed, suffix)
                if run_id:
                    file_stem = "{}-{}".format(file_stem, run_id)
                save_path = "{}/{}.pth".format(save_dir_path, file_stem)
                torch.save(model_state_dict, save_path)
                meta_path = "{}/{}.json".format(save_dir_path, file_stem)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(make_json_safe({
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "run": configs.get("run", {}),
                        "privacy": self.privacy_config,
                        "train": configs.get("train", {}),
                        "test": configs.get("test", {}),
                    }), f, indent=2, ensure_ascii=False)
                self.logger.log("Save model parameters to {}".format(save_path))
                self.logger.log("Save run metadata to {}".format(meta_path))
            else:
                save_dir_path = "./checkpoint/{}/tune".format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)

                now_para_str = configs["tune"]["now_para_str"]
                file_stem = "{}-{}-{}".format(model_name, now_para_str, suffix)
                if run_id:
                    file_stem = "{}-{}".format(file_stem, run_id)
                save_path = "{}/{}.pth".format(save_dir_path, file_stem)
                torch.save(model_state_dict, save_path)
                meta_path = "{}/{}.json".format(save_dir_path, file_stem)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(make_json_safe({
                        "model": model_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "run": configs.get("run", {}),
                        "privacy": self.privacy_config,
                        "tune": configs.get("tune", {}),
                    }), f, indent=2, ensure_ascii=False)
                self.logger.log("Save model parameters to {}".format(save_path))
                self.logger.log("Save run metadata to {}".format(meta_path))
