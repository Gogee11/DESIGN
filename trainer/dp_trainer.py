from copy import deepcopy
import math
import os
import re

import torch

from config.configurator import configs
from models.bulid_model import build_model
from trainer.trainer import Trainer
from trainer.utils import log_exceptions


class DPTrainer(Trainer):
    DP_TARGET_NAMES = {
        "user_embeds",
        "item_embeds",
        "user_embedding.weight",
        "item_embedding.weight",
    }

    def __init__(self, data_handler, logger):
        super().__init__(data_handler, logger)
        privacy_config = configs.get("privacy", {})

        self.privacy_config = {
            "enable": privacy_config.get("enable", True),
            "clip_norm": float(privacy_config.get("clip_norm", 1.0)),
            "noise_multiplier": float(privacy_config.get("noise_multiplier", 1.0)),
            "delta": privacy_config.get("delta"),
            "num_microbatches": int(privacy_config.get("num_microbatches", 1)),
            "target_scope": privacy_config.get("target_scope", "all"),
            "target_names": privacy_config.get("target_names"),
            "save_suffix": privacy_config.get("save_suffix", "dp"),
            "disable_model_randomness": privacy_config.get("disable_model_randomness", False),
        }

        if self.privacy_config["delta"] is None:
            train_size = max(len(self.data_handler.train_dataloader.dataset), 1)
            self.privacy_config["delta"] = 1.0 / train_size
        else:
            self.privacy_config["delta"] = float(self.privacy_config["delta"])

        self.dp_step_count = 0

    def _named_trainable_parameters(self, model):
        return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]

    def _split_parameters(self, model):
        target_scope = str(self.privacy_config.get("target_scope", "all")).lower()
        target_names = self.privacy_config.get("target_names")
        target_name_set = set(target_names) if isinstance(target_names, list) else self.DP_TARGET_NAMES

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

    def _create_grad_sums(self, named_parameters):
        return {name: torch.zeros_like(parameter) for name, parameter in named_parameters}

    def _clone_gradients(self, named_parameters):
        gradient_map = {}
        for name, parameter in named_parameters:
            if parameter.grad is None:
                gradient_map[name] = None
            else:
                gradient_map[name] = parameter.grad.detach().clone()
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

    def _zero_model_randomness(self, model):
        if not self.privacy_config["disable_model_randomness"]:
            return
        if hasattr(model, "keep_rate"):
            model.keep_rate = 1.0
        if hasattr(model, "mask_ratio"):
            model.mask_ratio = 0.0
        if hasattr(model, "is_training"):
            model.is_training = True

    def _slice_batch(self, batch_data, start, end):
        return [tensor[start:end] for tensor in batch_data]

    def _resolve_microbatch_count(self, batch_size):
        microbatch_count = int(self.privacy_config.get("num_microbatches", 1))
        microbatch_count = max(1, microbatch_count)
        return min(batch_size, microbatch_count)

    def _iter_microbatches(self, batch_data):
        batch_size = len(batch_data[0])
        microbatch_count = self._resolve_microbatch_count(batch_size)
        microbatch_size = math.ceil(batch_size / microbatch_count)
        for start in range(0, batch_size, microbatch_size):
            end = min(batch_size, start + microbatch_size)
            yield self._slice_batch(batch_data, start, end)

    def _accumulate_dp_gradients(self, dp_parameters, dp_grad_sums):
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
            dp_grad_sums[name].add_(parameter.grad.detach() * clip_coef)

        return grad_norm, clip_coef

    def _accumulate_normal_gradients(self, normal_parameters, normal_grad_sums):
        for name, parameter in normal_parameters:
            if parameter.grad is None:
                continue
            normal_grad_sums[name].add_(parameter.grad.detach())

    def _set_batch_gradients(self, dp_parameters, dp_grad_sums, normal_parameters, normal_gradients, microbatch_count):
        noise_scale = self.privacy_config["noise_multiplier"] * self.privacy_config["clip_norm"]

        for name, parameter in dp_parameters:
            noise = torch.randn_like(dp_grad_sums[name]) * noise_scale
            parameter.grad = (dp_grad_sums[name] + noise) / microbatch_count

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
            batch_size = len(batch_data[0])
            microbatch_count = self._resolve_microbatch_count(batch_size)
            dp_grad_sums = self._create_grad_sums(dp_parameters)
            batch_loss_dict = {}

            normal_gradients = {}
            if normal_parameters:
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss, batch_loss_dict = model.cal_loss(batch_data)
                batch_loss.backward()
                normal_gradients = self._clone_gradients(normal_parameters)
            else:
                batch_loss, batch_loss_dict = model.cal_loss(batch_data)

            self.optimizer.zero_grad(set_to_none=True)
            normal_grad_states = self._set_requires_grad(normal_parameters, False) if normal_parameters else {}
            for microbatch in self._iter_microbatches(batch_data):
                loss, _ = model.cal_loss(microbatch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._accumulate_dp_gradients(dp_parameters, dp_grad_sums)
            if normal_parameters:
                self._restore_requires_grad(normal_parameters, normal_grad_states)

            self.optimizer.zero_grad(set_to_none=True)
            self._set_batch_gradients(
                dp_parameters,
                dp_grad_sums,
                normal_parameters,
                normal_gradients,
                microbatch_count,
            )
            self.optimizer.step()
            self.dp_step_count += 1

            for loss_name, loss_value in batch_loss_dict.items():
                averaged_loss = float(loss_value) / len(train_dataloader)
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

        self.logger.log(
            "Enable handwritten DP-SGD training: noise_multiplier={}, clip_norm={}, delta={}, num_microbatches={}, target_scope={}, disable_model_randomness={}.".format(
                self.privacy_config["noise_multiplier"],
                self.privacy_config["clip_norm"],
                self.privacy_config["delta"],
                self.privacy_config["num_microbatches"],
                self.privacy_config["target_scope"],
                self.privacy_config["disable_model_randomness"],
            )
        )
        dp_parameters, normal_parameters = self._split_parameters(model)
        self.logger.log(
            "DP parameter split: dp_params={}, normal_params={}.".format(
                [name for name, _ in dp_parameters],
                len(normal_parameters),
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

        approx_epsilon = self._approximate_epsilon()
        self.save_model(model)
        self.logger.log(
            "Best Epoch {}. Final test result: {}. Handwritten DP-SGD config: {}. Approx privacy summary: (approx_epsilon={:.4f}, delta={}, steps={}).".format(
                best_epoch,
                test_result,
                self.privacy_config,
                approx_epsilon,
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
            suffix = self.privacy_config.get("save_suffix", "dp")

            if not configs["tune"]["enable"]:
                save_dir_path = "./checkpoint/{}".format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)

                save_path = "{}/{}-{}-{}-{}.pth".format(
                    save_dir_path,
                    model_name,
                    dataset_name,
                    seed,
                    suffix,
                )
                torch.save(model_state_dict, save_path)
                self.logger.log("Save model parameters to {}".format(save_path))
            else:
                save_dir_path = "./checkpoint/{}/tune".format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)

                now_para_str = configs["tune"]["now_para_str"]
                save_path = "{}/{}-{}-{}.pth".format(
                    save_dir_path,
                    model_name,
                    now_para_str,
                    suffix,
                )
                torch.save(model_state_dict, save_path)
                self.logger.log("Save model parameters to {}".format(save_path))
