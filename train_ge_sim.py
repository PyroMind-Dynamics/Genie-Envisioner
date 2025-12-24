import argparse
from pathlib import Path

from utils import import_custom_class


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GE-Sim (Cosmos2) model with Trainer runner."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="路径：Cosmos 配置文件，例如 configs/cosmos_model/acwm_cosmos.yaml",
    )
    parser.add_argument(
        "--runner_class_path",
        type=str,
        default="runner/ge_trainer.py",
        help="Trainer 类定义文件路径",
    )
    parser.add_argument(
        "--runner_class",
        type=str,
        default="Trainer",
        help="Trainer 类名",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help=(
            "可选：从已有 ckpt 继续训练（会覆盖 config 中 diffusion_model.model_path）。"
            "需确保 ckpt 与模型/配置兼容。"
        ),
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=None,
        help="可选：覆盖 config 中 steps_to_save，控制自动保存 ckpt 的步数间隔。",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="可选：覆盖 config 中 train_steps（总训练步数）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="可选：覆盖 config 中 output_dir。用于保存日志/ckpt。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    Runner = import_custom_class(args.runner_class, args.runner_class_path)

    # 初始化 Runner（Trainer）
    runner = Runner(args.config_file, output_dir=args.output_dir)

    # 覆盖训练步数/保存间隔
    if args.train_steps is not None:
        runner.args.train_steps = args.train_steps
    if args.save_every_steps is not None:
        runner.args.steps_to_save = args.save_every_steps

    # 覆盖初始权重（用于 warm start / 继续训练）
    if args.resume_checkpoint:
        runner.args.load_weights = True
        runner.args.load_diffusion_model_weights = True
        runner.args.diffusion_model["model_path"] = args.resume_checkpoint
        # 注意：需保证 resume_checkpoint 与当前 tokenizer/vae/transformer 配置兼容

    # 核心训练流程（与 main.py 的 train 分支一致）
    runner.prepare_dataset()
    runner.prepare_models()
    runner.prepare_trainable_parameters()
    runner.prepare_optimizer()
    runner.prepare_for_training()
    runner.prepare_trackers()
    runner.train()


if __name__ == "__main__":
    main()

