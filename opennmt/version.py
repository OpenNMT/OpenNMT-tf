"""OpenNMT-tf version."""

__version__ = "2.18.0"

INCLUSIVE_MIN_TF_VERSION = "2.3.0"
EXCLUSIVE_MAX_TF_VERSION = "2.5.0"


def _check_tf_version():
    from distutils.version import LooseVersion

    import tensorflow as tf
    import warnings

    if (
        LooseVersion(INCLUSIVE_MIN_TF_VERSION)
        <= LooseVersion(tf.__version__)
        < LooseVersion(EXCLUSIVE_MAX_TF_VERSION)
    ):
        return

    warnings.warn(
        "OpenNMT-tf supports TensorFlow versions %s (included) to %s (excluded), "
        "but you have TensorFlow %s installed. Some features might not work properly."
        % (INCLUSIVE_MIN_TF_VERSION, EXCLUSIVE_MAX_TF_VERSION, tf.__version__),
        UserWarning,
    )
