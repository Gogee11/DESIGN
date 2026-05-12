from copy import deepcopy
import math
import os

import torch

from config.configurator import configs
from models.bulid_model import build_model
from trainer.trainer import Trainer
from trainer.utils import log_exceptions


class DPTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super().__init__(data_handler, logger)
        privacy_config = configs.get("privacy", {})

        self.privacy_config = {
            "enable": privacy_config.get("enable", True),
            "clip_norm": privacy_config.get("clip_norm", 1.0),
            "noise_multiplier": privacy_config.get("noise_multiplier", 1.0),
            "delta": privacy_config.get("delta"),
            "save_suffix": privacy_config.get("save_suffix", "dp"),
            "disable_model_randomness": privacy_config.get("disable_model_randomness", False),
        }

        if self.privacy_config["delta"] is None:
            train_size = max(len(self.data_handler.train_dataloader.dataset), 1)
            self.privacy_config["delta"] = 1.0 / train_size

        self.dp_step_count = 0

    def _trainable_parameters(self, model):
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    def _create_grad_sums(self, model):
        grad_sums = []
        for parameter in self._trainable_parameters(model):
            grad_sums.append(torch.zeros_like(parameter))
        return grad_sums

    def _zero_model_randomness(self, model):
        if not self.privacy_config["disable_model_randomness"]:
            return
        if hasattr(model, "keep_rate"):
            model.keep_rate = 1.0
        if hasattr(model, "mask_ratio"):
            model.mask_ratio = 0.0
        if hasattr(model, "is_training"):
            model.is_training = True

    def _single_example_batch(self, batch_data, index):
        return [tensor[index:index + 1] for tensor in batch_data]

    def _accumulate_clipped_gradients(self, model, grad_sums):
        trainable_parameters = self._trainable_parameters(model)
        grad_norm_sq = torch.zeros(1, device=configs["device"])

        for parameter in trainable_parameters:
            if parameter.grad is None:
                continue
            grad_norm_sq = grad_norm_sq + parameter.grad.detach().pow(2).sum()

        grad_norm = torch.sqrt(grad_norm_sq).item()
        clip_norm = self.privacy_config["clip_norm"]
        clip_coef = min(1.0, clip_norm / (grad_norm + 1e-12))

        for grad_sum, parameter in zip(grad_sums, trainable_parameters):
            if parameter.grad is None:
                continue
            grad_sum.add_(parameter.grad.detach() * clip_coef)

        return grad_norm, clip_coef

    def _set_noisy_gradients(self, model, grad_sums, batch_size):
        trainable_parameters = self._trainable_parameters(model)
        noise_scale = self.privacy_config["noise_multiplier"] * self.privacy_config["clip_norm"]

        for grad_sum, parameter in zip(grad_sums, trainable_parameters):
            noise = torch.randn_like(grad_sum) * noise_scale
            parameter.grad = (grad_sum + noise) / batch_size

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

        for _, tem in enumerate(train_dataloader):
            batch_data = [tensor.long().to(configs["device"]) for tensor in tem]
            batch_size = len(batch_data[0])
            grad_sums = self._create_grad_sums(model)
            batch_loss_dict = {}

            for sample_index in range(batch_size):
                self.optimizer.zero_grad(set_to_none=True)
                single_batch = self._single_example_batch(batch_data, sample_index)
                loss, loss_dict = model.cal_loss(single_batch)
                loss.backward()
                self._accumulate_clipped_gradients(model, grad_sums)

                for loss_name, loss_value in loss_dict.items():
                    batch_loss_dict[loss_name] = batch_loss_dict.get(loss_name, 0.0) + float(loss_value)

            self.optimizer.zero_grad(set_to_none=True)
            self._set_noisy_gradients(model, grad_sums, batch_size)
            self.optimizer.step()
            self.dp_step_count += 1

            for loss_name, total_loss_value in batch_loss_dict.items():
                averaged_loss = total_loss_value / batch_size / len(train_dataloader)
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
            "Enable handwritten DP-SGD: noise_multiplier={}, clip_norm={}, delta={}, disable_model_randomness={}.".format(
                self.privacy_config["noise_multiplier"],
                self.privacy_config["clip_norm"],
                self.privacy_config["delta"],
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

        approx_epsilon = self._approximate_epsilon()
        self.save_model(model)
        self.logger.log(
            "Best Epoch {}. Final test result: {}. Handwritten DP config: {}. Approx privacy summary: (approx_epsilon={:.4f}, delta={}, steps={}).".format(
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
