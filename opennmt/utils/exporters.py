"""Define model exporters."""

import abc
import os
import tempfile

import tensorflow as tf

from opennmt.utils import misc


class Exporter(abc.ABC):
    """Base class for model exporters."""

    def export(self, model, export_dir):
        """Exports :obj:`model` to :obj:`export_dir`.

        Raises:
          ValueError: if :obj:`model` is not supported by this exporter.
        """
        self._export_model(model, export_dir)
        with tempfile.TemporaryDirectory() as tmp_dir:
            extra_assets = model.export_assets(tmp_dir)
            if extra_assets:
                assets_extra = os.path.join(export_dir, "assets.extra")
                tf.io.gfile.makedirs(assets_extra)
                for filename, path in extra_assets.items():
                    tf.io.gfile.copy(
                        path, os.path.join(assets_extra, filename), overwrite=True
                    )
                tf.get_logger().info("Extra assets written to: %s", assets_extra)

    @abc.abstractmethod
    def _export_model(self, model, export_dir):
        raise NotImplementedError()


_EXPORTERS_REGISTRY = misc.ClassRegistry(base_class=Exporter)
register_exporter = _EXPORTERS_REGISTRY.register


def make_exporter(name, **kwargs):
    """Creates a new exporter.

    Args:
      name: The exporter name.
      **kwargs: Additional arguments to pass to the exporter constructor.

    Returns:
      A :class:`opennmt.utils.Exporter` instance.

    Raises:
      ValueError: if :obj:`name` is invalid.
    """
    exporter_class = _EXPORTERS_REGISTRY.get(name)
    if exporter_class is None:
        raise ValueError("Invalid exporter name: %s" % name)
    return exporter_class(**kwargs)


def list_exporters():
    """Lists the name of registered exporters."""
    return _EXPORTERS_REGISTRY.class_names


@register_exporter(name="saved_model")
class SavedModelExporter(Exporter):
    """SavedModel exporter."""

    def _export_model(self, model, export_dir):
        tf.saved_model.save(model, export_dir, signatures=model.serve_function())


@register_exporter(name="checkpoint")
class CheckpointExporter(Exporter):
    """Checkpoint exporter."""

    def _export_model(self, model, export_dir):
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.write(os.path.join(export_dir, "ckpt"))


@register_exporter(name="tflite")
class TFLiteExporter(Exporter):
    """TensorFlow Lite exporter."""

    def __init__(self, quantization=None):
        accepted_quantization = ("float16",)
        if quantization is not None and quantization not in accepted_quantization:
            raise ValueError(
                "Unsupported quantization '%s' for TensorFlow Lite, accepted values are: %s"
                % (quantization, ", ".join(accepted_quantization))
            )
        self._quantization = quantization

    def _export_model(self, model, export_dir):
        if not model.built:
            model.create_variables()

        # Tries to run prediction with TensorFlow Lite method it will convert
        tflite_concrete_fn = model.tflite_function().get_concrete_function()

        # Saving
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [tflite_concrete_fn]
        )
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if self._quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]

        tflite_model_path = os.path.join(export_dir, "opennmt.tflite")
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(tflite_model_path, "wb") as f:
            f.write(tflite_model)


@register_exporter(name="tflite_float16")
class TFLiteFloat16Exporter(TFLiteExporter):
    """TensorFlow Lite exporter with float16 quantization."""

    def __init__(self):
        super().__init__(quantization="float16")


@register_exporter(name="ctranslate2")
class CTranslate2Exporter(Exporter):
    """CTranslate2 exporter."""

    def __init__(self, quantization=None):
        """Initializes the exporter.

        Args:
          quantization: Quantize model weights to this type when exporting the model.
            Can be "int8", "int16", or "float16". Default is no quantization.

        Raises:
          ImportError: if the CTranslate2 package is missing.
          ValueError: if :obj:`quantization` is invalid.
        """
        # Fail now if ctranslate2 package is missing.
        import ctranslate2

        accepted_quantization = ("int8", "int16", "float16")
        if quantization is not None and quantization not in accepted_quantization:
            raise ValueError(
                "Invalid quantization '%s' for CTranslate2, accepted values are: %s"
                % (quantization, ", ".join(accepted_quantization))
            )
        self._quantization = quantization

    def _export_model(self, model, export_dir):
        model_spec = model.ctranslate2_spec
        if model_spec is None:
            raise ValueError(
                "The model does not define an equivalent CTranslate2 model specification"
            )
        if not model.built:
            model.create_variables()
        variables = misc.get_variables_name_mapping(model, root_key="model")
        variables = {
            name.replace("/.ATTRIBUTES/VARIABLE_VALUE", ""): value.numpy()
            for name, value in variables.items()
        }
        import ctranslate2

        converter = ctranslate2.converters.OpenNMTTFConverter(
            src_vocab=model.features_inputter.vocabulary_file,
            tgt_vocab=model.labels_inputter.vocabulary_file,
            variables=variables,
        )
        converter.convert(
            export_dir, model_spec, quantization=self._quantization, force=True
        )


@register_exporter(name="ctranslate2_int8")
class CTranslate2Int8Exporter(CTranslate2Exporter):
    """CTranslate2 exporter with int8 quantization."""

    def __init__(self):
        super().__init__(quantization="int8")


@register_exporter(name="ctranslate2_int16")
class CTranslate2Int16Exporter(CTranslate2Exporter):
    """CTranslate2 exporter with int16 quantization."""

    def __init__(self):
        super().__init__(quantization="int16")


@register_exporter(name="ctranslate2_float16")
class CTranslate2Float16Exporter(CTranslate2Exporter):
    """CTranslate2 exporter with float16 quantization."""

    def __init__(self):
        super().__init__(quantization="float16")
