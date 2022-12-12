"""OpenNMT-tf version."""

__version__ = "2.30.0"

INCLUSIVE_MIN_TF_VERSION = "2.6.0"
EXCLUSIVE_MAX_TF_VERSION = "2.12.0"


def _check_tf_version():
    import warnings

    import tensorflow as tf

    from packaging.version import Version

    if (
        Version(INCLUSIVE_MIN_TF_VERSION)
        <= Version(tf.__version__)
        < Version(EXCLUSIVE_MAX_TF_VERSION)
    ):
        return

    warnings.warn(
        "OpenNMT-tf supports TensorFlow versions %s (included) to %s (excluded), "
        "but you have TensorFlow %s installed. Some features might not work properly."
        % (INCLUSIVE_MIN_TF_VERSION, EXCLUSIVE_MAX_TF_VERSION, tf.__version__),
        UserWarning,
    )
