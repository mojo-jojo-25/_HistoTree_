import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import time
import torch
import numpy as np
from HistoTree import HistoTree
import random


torch.set_printoptions(sci_mode=False)
class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=globals()["__doc__"])

        self.parser.add_argument(
            "--config", type=str, required=True, help="Path to the config file"
        )
        self.parser.add_argument('--device', type=int, default=0, help='GPU device id')
        self.parser.add_argument("--seed", type=int, default=12321, help="Random seed")
        self.parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
        self.parser.add_argument("--feature_path", type=str, default=None, help="This argument will overwrite the dataroot in the config if it is not None.")
        self.parser.add_argument("--verbose",type=str,default="info",help="Verbose level: info | debug | warning | critical")
        self.parser.add_argument("--eval_best",action="store_true",help="Evaluate by best model during training, instead of the ckpt stored at the last epoch")
        self.parser.add_argument("--ni",action="store_true",help="No interaction. Suitable for Slurm Job launcher",)
        self.parser.add_argument('--n_class', type=int, default=2, help='classification classes')
        self.parser.add_argument('--model_path', type=str, help='path to trained model')
        self.parser.add_argument('--log_path', type=str, help='path to log files')
        self.parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        self.parser.add_argument('--train', action='store_true', default=False, help='train only')
        self.parser.add_argument('--test', action='store_true', default=False, help='test only')
        self.parser.add_argument('--explain', action='store_true', default=False, help='explain')
        self.parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('--train_set', type=str, default = 'lung_files/train_0.txt')
        self.parser.add_argument('--val_set', type=str, default = 'lung_files/val_0.txt')
        self.parser.add_argument('--depth', type=int, default=3, help ='depth')
        self.parser.add_argument('--vis_folder', type=str, default ='test')
        self.parser.add_argument('--n_epochs', type=int, default = 50)
        self.parser.add_argument('--dataset', type=str, help ='path to log files')
        self.parser.add_argument('--task', type=str, help ='path to log files', default='classification')

    def parse(self):
        args = self.parser.parse_args()

        return args

args = Options().parse()
def parse_config():

    args.log_path = os.path.join(args.exp, "logs")

    # parse config file
    with open(os.path.join(args.config), "r") as f:
            config = yaml.safe_load(f)
            new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard-{}".format(args.task_name))
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    if not args.ni:
        import torch.utils.tensorboard as tb

    if not args.depth is None:
        new_config.tree.depth = args.depth

    if not args.feature_path is None:
        new_config.data.feature_path = args.feature_path


    # if not args.test and not args.explain:
    #     if not args.resume_training:
    #         if os.path.exists(args.log_path):
    #             overwrite = False
    #             if args.ni:
    #                 overwrite = True
    #             else:
    #                 response = input("Folder already exists. Overwrite? (Y/N)")
    #                 if response.upper() == "Y":
    #                     overwrite = True
    #
    #             if overwrite:
    #                 shutil.rmtree(args.log_path)
    #                 shutil.rmtree(tb_path)
    #                 os.makedirs(args.log_path)
    #                 if os.path.exists(tb_path):
    #                     shutil.rmtree(tb_path)
    #             else:
    #                 print("Folder exists. Program halted.")
    #                 sys.exit(0)
    #         else:
    #             os.makedirs(args.log_path)


        if not args.ni:
            new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        else:
            new_config.tb_logger = None
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "{}.txt".format(args.task_name)))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:

        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        # saving test metrics to a .txt file
        handler2 = logging.FileHandler(os.path.join(args.log_path, "testmetrics.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    return new_config, logger

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def seed_torch(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():

    config, logger = parse_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))

    device_name = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logging.info("Using device: {}".format(device))
    config.device = device

    try:
        runner = HistoTree(args, config, device=config.device)
        start_time = time.time()
        procedure = None

        seed_torch(args.seed,  config.device)

        if args.test:
            procedure = "Testing"
            config.training.n_epochs = 1
            runner.training()
        elif args.explain:
            procedure = "Explain"
            config.training.n_epochs = 1
            runner.explain()
        else:
            procedure = "Training"
            runner.training()

        end_time = time.time()
        logging.info("\n{} procedure finished. It took {:.4f} minutes.\n\n\n".format(
            procedure, (end_time - start_time) / 60))

    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":

    sys.exit(main())
