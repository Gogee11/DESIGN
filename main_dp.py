import datetime
import logging
import os

from config.configurator import configs
from load_data.build_data_handler import build_data_handler
from models.bulid_model import build_model
from trainer.dp_trainer import DPTrainer
from trainer.logger import Logger
from trainer.utils import set_seed


class DPLogger(Logger):
    def __init__(self, log_configs=True):
        model_name = configs["model"]["name"]
        dataset_name = configs["data"]["name"]
        suffix = configs.get("privacy", {}).get("save_suffix", "dp")
        log_dir_path = "./log/{}".format(model_name)

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

        self.logger = logging.getLogger("train_logger_dp_{}_{}_{}".format(model_name, dataset_name, suffix))
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False

        timestamp = datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
        if not configs["tune"]["enable"]:
            log_path = "{}/{}_{}_{}.log".format(log_dir_path, dataset_name, suffix, timestamp)
        else:
            log_path = "{}/{}-{}-tune_{}.log".format(log_dir_path, dataset_name, suffix, timestamp)

        log_file = logging.FileHandler(log_path, "a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)

        if log_configs:
            tmp_configs = {}
            tmp_configs["optimizer"] = configs["optimizer"]
            tmp_configs["train"] = configs["train"]
            tmp_configs["test"] = configs["test"]
            tmp_configs["data"] = configs["data"]
            tmp_configs["model"] = configs["model"]
            tmp_configs["privacy"] = configs.get("privacy", {})
            self.log(tmp_configs)


if __name__ == "__main__":
    set_seed(configs["train"]["seed"])

    data_handler = build_data_handler()
    data_handler.load_data()

    model = build_model(data_handler).to(configs["device"])

    logger = DPLogger()
    trainer = DPTrainer(data_handler, logger)
    trainer.train(model)
