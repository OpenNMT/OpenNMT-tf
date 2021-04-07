"""Main script."""

import argparse
import logging
import os
import sys

import tensorflow as tf

from opennmt import __version__
from opennmt.models import catalog
from opennmt.runner import Runner
from opennmt.config import load_model, load_config
from opennmt.utils import exporters


_PYTHON_TO_TENSORFLOW_LOGGING_LEVEL = {
    logging.CRITICAL: 3,
    logging.ERROR: 2,
    logging.WARNING: 1,
    logging.INFO: 0,
    logging.DEBUG: 0,
    logging.NOTSET: 0,
}


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

    # Align the TensorFlow C++ log level with the Python level.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(
        _PYTHON_TO_TENSORFLOW_LOGGING_LEVEL[log_level]
    )


def _prefix_paths(prefix, paths):
    """Recursively prefix paths.

    Args:
      prefix: The prefix to apply.
      data: A dict of relative paths.

    Returns:
      The updated dict.
    """
    if isinstance(paths, dict):
        for key, path in paths.items():
            paths[key] = _prefix_paths(prefix, path)
        return paths
    elif isinstance(paths, list):
        for i, path in enumerate(paths):
            paths[i] = _prefix_paths(prefix, path)
        return paths
    elif isinstance(paths, str):
        path = paths
        new_path = os.path.join(prefix, path)
        if tf.io.gfile.exists(new_path):
            return new_path
        else:
            return path
    else:
        return paths


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
        "--auto_config",
        default=False,
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
            "Specific checkpoint or model directory to load "
            "(when a directory is set, the latest checkpoint is used)."
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
    config = load_config(args.config)
    if args.run_dir:
        config["model_dir"] = os.path.join(args.run_dir, config["model_dir"])
    if args.data_dir:
        config["data"] = _prefix_paths(args.data_dir, config["data"])

    if is_master and not tf.io.gfile.exists(config["model_dir"]):
        tf.get_logger().info("Creating model directory %s", config["model_dir"])
        tf.io.gfile.makedirs(config["model_dir"])

    model = load_model(
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
        )
    elif args.run_type == "average_checkpoints":
        runner.average_checkpoints(args.output_dir, max_count=args.max_count)
    elif args.run_type == "update_vocab":
        runner.update_vocab(
            args.output_dir, src_vocab=args.src_vocab, tgt_vocab=args.tgt_vocab
        )


if __name__ == "__main__":
    main()
