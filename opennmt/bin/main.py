"""Main script."""

import argparse
import logging
import os
import sys

import tensorflow as tf

from opennmt import __version__
from opennmt import config as config_util
from opennmt.models import catalog
from opennmt.runner import Runner
from opennmt.utils import exporters


def _initialize_logging(log_level):
    logger = tf.get_logger()
    logger.setLevel(log_level)

    # Configure the TensorFlow logger to use the same log format as the TensorFlow C++ logs.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d000: %(levelname).1s %(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-v", "--version", action="version", version="OpenNMT-tf %s" % __version__
    )
    parser.add_argument(
        "--config", required=True, nargs="+", help="List of configuration files."
    )
    parser.add_argument(
        "--model_dir",
        help=(
            "Path to the model directory. If not set, the model directory is read "
            "from the field 'model_dir' in the configuration."
        ),
    )
    parser.add_argument(
        "--auto_config",
        default=None,
        action="store_true",
        help="Enable automatic configuration values.",
    )
    parser.add_argument(
        "--model_type",
        default="",
        choices=list(sorted(catalog.list_model_names_from_catalog())),
        help="Model type from the catalog.",
    )
    parser.add_argument("--model", default="", help="Custom model configuration file.")
    parser.add_argument(
        "--run_dir",
        default="",
        help="If set, model_dir will be created relative to this location.",
    )
    parser.add_argument(
        "--data_dir",
        default="",
        help="If set, data files are expected to be relative to this location.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help=(
            "Path to the checkpoint or checkpoint directory to load. "
            "If not set, the latest checkpoint from the model directory is loaded."
        ),
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Logs verbosity.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--gpu_allow_growth",
        default=False,
        action="store_true",
        help="Allocate GPU memory dynamically.",
    )
    parser.add_argument(
        "--intra_op_parallelism_threads",
        type=int,
        default=0,
        help=(
            "Number of intra op threads (0 means the system picks "
            "an appropriate number)."
        ),
    )
    parser.add_argument(
        "--inter_op_parallelism_threads",
        type=int,
        default=0,
        help=(
            "Number of inter op threads (0 means the system picks "
            "an appropriate number)."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        default=False,
        action="store_true",
        help="Enable mixed precision.",
    )
    parser.add_argument(
        "--eager_execution",
        default=False,
        action="store_true",
        help="Enable TensorFlow eager execution.",
    )

    subparsers = parser.add_subparsers(help="Run type.", dest="run_type")
    subparsers.required = True
    parser_train = subparsers.add_parser("train", help="Training.")
    parser_train.add_argument(
        "--with_eval",
        default=False,
        action="store_true",
        help="Enable automatic evaluation.",
    )
    parser_train.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for in-graph replication.",
    )
    parser_train.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Enable Horovod training mode.",
    )
    parser_train.add_argument(
        "--continue_from_checkpoint",
        default=False,
        action="store_true",
        help=(
            "Continue the training from the checkpoint passed to --checkpoint_path. "
            "If --checkpoint_path is set but not --continue_from_checkpoint, only the "
            "model weights are loaded and the optimization states are ignored."
        ),
    )

    parser_eval = subparsers.add_parser("eval", help="Evaluation.")
    parser_eval.add_argument(
        "--features_file", nargs="+", default=None, help="Input features files."
    )
    parser_eval.add_argument("--labels_file", default=None, help="Output labels files.")

    parser_infer = subparsers.add_parser("infer", help="Inference.")
    parser_infer.add_argument(
        "--features_file", nargs="+", required=True, help="Run inference on this file."
    )
    parser_infer.add_argument(
        "--predictions_file",
        default="",
        help=(
            "File used to save predictions. If not set, predictions are printed "
            "on the standard output."
        ),
    )
    parser_infer.add_argument(
        "--log_prediction_time",
        default=False,
        action="store_true",
        help="Logs some prediction time metrics.",
    )

    parser_export = subparsers.add_parser("export", help="Model export.")
    parser_export.add_argument(
        "--output_dir",
        "--export_dir",
        required=True,
        help="The directory of the exported model.",
    )
    parser_export.add_argument(
        "--format",
        "--export_format",
        choices=exporters.list_exporters(),
        default="saved_model",
        help="Format of the exported model.",
    )

    parser_score = subparsers.add_parser("score", help="Scoring.")
    parser_score.add_argument(
        "--features_file", nargs="+", required=True, help="Features file."
    )
    parser_score.add_argument(
        "--predictions_file", default=None, help="Predictions to score."
    )
    parser_score.add_argument("--output_file", default=None, help="Output file.")

    parser_average_checkpoints = subparsers.add_parser(
        "average_checkpoints", help="Checkpoint averaging."
    )
    parser_average_checkpoints.add_argument(
        "--output_dir",
        required=True,
        help="The output directory for the averaged checkpoint.",
    )
    parser_average_checkpoints.add_argument(
        "--max_count",
        type=int,
        default=8,
        help="The maximal number of checkpoints to average.",
    )

    parser_update_vocab = subparsers.add_parser(
        "update_vocab", help="Update model vocabularies in checkpoint."
    )
    parser_update_vocab.add_argument(
        "--output_dir",
        required=True,
        help="The output directory for the updated checkpoint.",
    )
    parser_update_vocab.add_argument(
        "--src_vocab", default=None, help="Path to the new source vocabulary."
    )
    parser_update_vocab.add_argument(
        "--tgt_vocab", default=None, help="Path to the new target vocabulary."
    )

    # When using an option that takes multiple values just before the run type,
    # the run type is treated as a value of this option. To fix this issue, we
    # inject a placeholder option just before the run type to clearly separate it.
    parser.add_argument("--placeholder", action="store_true", help=argparse.SUPPRESS)
    run_types = set(subparsers.choices.keys())
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in run_types:
            args.insert(i, "--placeholder")
            break

    args = parser.parse_args(args)
    if (
        hasattr(args, "features_file")
        and args.features_file
        and len(args.features_file) == 1
    ):
        args.features_file = args.features_file[0]

    _initialize_logging(getattr(logging, args.log_level))
    tf.config.threading.set_intra_op_parallelism_threads(
        args.intra_op_parallelism_threads
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        args.inter_op_parallelism_threads
    )

    if args.eager_execution:
        tf.config.run_functions_eagerly(True)

    gpus = tf.config.list_physical_devices(device_type="GPU")
    if hasattr(args, "horovod") and args.horovod:
        import horovod.tensorflow as hvd

        hvd.init()
        is_master = hvd.rank() == 0
        if gpus:
            local_gpu = gpus[hvd.local_rank()]
            tf.config.set_visible_devices(local_gpu, device_type="GPU")
            gpus = [local_gpu]
    else:
        hvd = None
        is_master = True

    if args.gpu_allow_growth:
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, enable=True)

    # Load and merge run configurations.
    config = config_util.load_config(args.config)
    if args.model_dir:
        config["model_dir"] = args.model_dir
    elif "model_dir" not in config:
        raise ValueError(
            "No model directory is defined: you should either set --model_dir "
            "on the command line or set the field model_dir in the configuration"
        )
    if args.run_dir:
        config["model_dir"] = os.path.join(args.run_dir, config["model_dir"])
    if args.data_dir:
        config["data"] = config_util.try_prefix_paths(args.data_dir, config["data"])

    if is_master and not tf.io.gfile.exists(config["model_dir"]):
        tf.get_logger().info("Creating model directory %s", config["model_dir"])
        tf.io.gfile.makedirs(config["model_dir"])

    model = config_util.load_model(
        config["model_dir"],
        model_file=args.model,
        model_name=args.model_type,
        serialize_model=is_master,
        as_builder=True,
    )
    runner = Runner(
        model,
        config,
        auto_config=args.auto_config,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )

    if args.run_type == "train":
        runner.train(
            num_devices=args.num_gpus,
            with_eval=args.with_eval,
            checkpoint_path=args.checkpoint_path,
            hvd=hvd,
            continue_from_checkpoint=args.continue_from_checkpoint,
        )
    elif args.run_type == "eval":
        metrics = runner.evaluate(
            checkpoint_path=args.checkpoint_path,
            features_file=args.features_file,
            labels_file=args.labels_file,
        )
        print(metrics)
    elif args.run_type == "infer":
        runner.infer(
            args.features_file,
            predictions_file=args.predictions_file,
            checkpoint_path=args.checkpoint_path,
            log_time=args.log_prediction_time,
        )
    elif args.run_type == "export":
        runner.export(
            args.output_dir,
            checkpoint_path=args.checkpoint_path,
            exporter=exporters.make_exporter(args.format),
        )
    elif args.run_type == "score":
        runner.score(
            args.features_file,
            args.predictions_file,
            checkpoint_path=args.checkpoint_path,
            output_file=args.output_file,
        )
    elif args.run_type == "average_checkpoints":
        runner.average_checkpoints(args.output_dir, max_count=args.max_count)
    elif args.run_type == "update_vocab":
        runner.update_vocab(
            args.output_dir, src_vocab=args.src_vocab, tgt_vocab=args.tgt_vocab
        )


if __name__ == "__main__":
    main()
