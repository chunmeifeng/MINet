"""

Chun-Mei Feng, Harbin Institute of Technology, Shenzhen / Inception Institute of Artifical Intelligence,UAE

Our code bulid on fastMRI https://github.com/facebookresearch/fastMRI

"""

import pathlib
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything


sys.path.append('/home/chunmeifeng/MINet/')

from fastmri.data.mri_data import fetch_dir
from module_MINet import SRModule 


def main(args):
    """Main training routine."""
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    seed_everything(args.seed)
    model = SRModule(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING OR TEST
    # ------------------------
    if args.mode == "train":
        trainer.fit(model)
    elif args.mode == "test":
        assert args.resume_from_checkpoint is not None
        trainer.test(model)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    path_config = pathlib.Path.cwd() / "mriSR_dirs.yaml"
    knee_path = fetch_dir("knee_path", path_config)
    logdir = fetch_dir("log_path", path_config) / "minet" / "2x_SR"


    parent_parser = ArgumentParser(add_help=False)

    parser = SRModule.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)

    num_gpus = 1
    backend = "ddp"
    batch_size = 4 if backend == "ddp" else num_gpus

    # module config
    config = dict(
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        mask_type="random",
        center_fractions=[0.08],
        accelerations=[4],
        n_channels_in=1,
        n_channels_out=1,
        n_resgroups = 6,    
        n_resblocks = 6,    
        n_feats = 64, 
        lr=0.00001,     
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        data_path=knee_path,
        challenge="singlecoil",
        exp_dir=logdir,
        exp_name="unet_demo",
        test_split="test",
        batch_size=batch_size,
    )
    parser.set_defaults(**config)

    # trainer config
    parser.set_defaults(
        gpus=num_gpus,
        max_epochs=35,
        default_root_dir=logdir,
        replace_sampler_ddp=(backend != "ddp"),
        distributed_backend=backend,
        seed=42,
        deterministic=True,
        # resume_from_checkpoint = '/checkpoints/epoch=34.ckpt'
    )


    parser.add_argument("--mode", default="train", type=str)
    args = parser.parse_args()

    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
