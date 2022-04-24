import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch

from utils import save_config, load_config
from evaluation import evalrank
from co_train import main


def run():

    # current_time
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    # Hyper Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    parser.add_argument(
        "--data_path", default="/data/NCR-data/data", help="path to datasets"
    )
    parser.add_argument(
        "--data_name", default="f30k_precomp", help="{coco,f30k,cc152k}_precomp"
    )
    parser.add_argument(
        "--vocab_path",
        default="/data/NCR-data/vocab",
        help="Path to saved vocabulary json files.",
    )

    # ----------------------- training setting ----------------------#
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Size of a training mini-batch."
    )
    parser.add_argument(
        "--num_epochs", default=40, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr_update",
        default=30,
        type=int,
        help="Number of epochs to update the learning rate.",
    )
    parser.add_argument(
        "--learning_rate", default=0.0002, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--workers", default=0, type=int, help="Number of data loader workers."
    )
    parser.add_argument(
        "--log_step",
        default=100,
        type=int,
        help="Number of steps to print and record the log.",
    )
    parser.add_argument(
        "--grad_clip", default=2.0, type=float, help="Gradient clipping threshold."
    )
    parser.add_argument("--margin", default=0.2, type=float, help="Rank loss margin.")

    # ------------------------- model setting -----------------------#
    parser.add_argument(
        "--img_dim",
        default=2048,
        type=int,
        help="Dimensionality of the image embedding.",
    )
    parser.add_argument(
        "--word_dim",
        default=300,
        type=int,
        help="Dimensionality of the word embedding.",
    )
    parser.add_argument(
        "--embed_size",
        default=1024,
        type=int,
        help="Dimensionality of the joint embedding.",
    )
    parser.add_argument(
        "--sim_dim", default=256, type=int, help="Dimensionality of the sim embedding."
    )
    parser.add_argument(
        "--num_layers", default=1, type=int, help="Number of GRU layers."
    )
    parser.add_argument("--bi_gru", action="store_false", help="Use bidirectional GRU.")
    parser.add_argument(
        "--no_imgnorm",
        action="store_true",
        help="Do not normalize the image embeddings.",
    )
    parser.add_argument(
        "--no_txtnorm",
        action="store_true",
        help="Do not normalize the text embeddings.",
    )
    parser.add_argument("--module_name", default="SGR", type=str, help="SGR, SAF")
    parser.add_argument("--sgr_step", default=3, type=int, help="Step of the SGR.")

    # noise settings
    parser.add_argument("--noise_file", default="", help="noise_file")
    parser.add_argument("--noise_ratio", default=0.2, type=float, help="Noisy ratio")

    # NCR Settings
    parser.add_argument(
        "--no_co_training", action="store_true", help="No co-training for noisy label."
    )
    parser.add_argument("--warmup_epoch", default=1, type=int, help="warm up epochs")
    parser.add_argument("--warmup_model_path", default="", help="warm up models")
    parser.add_argument(
        "--p_threshold", default=0.5, type=float, help="clean probability threshold"
    )
    parser.add_argument(
        "--soft_margin", default="exponential", help="linear|exponential|sin"
    )

    # Runing Settings
    parser.add_argument("--gpu", default="0", help="Which gpu to use.")
    parser.add_argument(
        "--seed", default=random.randint(0, 100), type=int, help="Random seed."
    )
    parser.add_argument(
        "--output_dir", default=os.path.join("output", current_time), help="Output dir."
    )

    # load arguments
    opt = parser.parse_args()

    # Output dir
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    if not opt.noise_file:
        opt.noise_file = os.path.join(
            opt.output_dir, opt.data_name + "_" + str(opt.noise_ratio) + ".npy"
        )

    if opt.data_name == "cc152k_precomp":
        opt.noise_ratio = 0
        opt.noise_file = ""

    print("\n*-------- Experiment Config --------*")
    print(opt)

    # CUDA env
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    # set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.random.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True

    # save config
    save_config(opt, os.path.join(opt.output_dir, "config.json"))

    # traing and evaluation
    print("\n*-------- Training --------*")
    main(opt)

    print("\n*-------- Testing --------*")
    if opt.data_name == "coco_precomp":
        print("5 fold validation")
        evalrank(
            os.path.join(opt.output_dir, "model_best.pth.tar"),
            split="testall",
            fold5=True,
        )
        print("full validation")
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="testall")
    else:
        evalrank(os.path.join(opt.output_dir, "model_best.pth.tar"), split="test")


if __name__ == "__main__":
    run()
