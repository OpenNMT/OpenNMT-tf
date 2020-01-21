"""Define model exporters."""

import abc
import os
import tempfile

import tensorflow as tf

from opennmt.utils import misc


def make_exporter(name, **kwargs):
  """Creates a new exporter.

  Args:
    name: The exporter name, can be "saved_model", "ctranslate2".
    **kwargs: Additional arguments to pass to the exporter constructor.

  Returns:
    A :class:`opennmt.utils.Exporter` instance.

  Raises:
    ValueError: if :obj:`name` is invalid.
  """
  if name == "saved_model":
    return SavedModelExporter(**kwargs)
  elif name == "ctranslate2":
    return CTranslate2Exporter(**kwargs)
  else:
    raise ValueError("Invalid exporter name: %s" % name)


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
          tf.io.gfile.copy(path, os.path.join(assets_extra, filename), overwrite=True)
        tf.get_logger().info("Extra assets written to: %s", assets_extra)

  @abc.abstractmethod
  def _export_model(self, model, export_dir):
    raise NotImplementedError()


class SavedModelExporter(Exporter):
  """SavedModel exporter."""

  def _export_model(self, model, export_dir):
    tf.saved_model.save(model, export_dir, signatures=model.serve_function())


class CTranslate2Exporter(Exporter):
  """CTranslate2 exporter."""

  def __init__(self, quantization=None):
    """Initializes the exporter.

    Args:
      quantization: Quantize model weights to this type when exporting the model.
        Can be "int16" or "int8". Default is no quantization.
    """
    # Fail now if ctranslate2 package is missing.
    import ctranslate2  # pylint: disable=import-outside-toplevel,unused-import
    self._quantization = quantization

  def _export_model(self, model, export_dir):
    model_spec = model.ctranslate2_spec
    if model_spec is None:
      raise ValueError("The model does not define an equivalent CTranslate2 model specification")
    if not model.built:
      model.create_variables()
    _, variables = misc.get_variables_name_mapping(model, root_key="model")
    variables = {
        name.replace("/.ATTRIBUTES/VARIABLE_VALUE", ""):value.numpy()
        for name, value in variables.items()}
    import ctranslate2  # pylint: disable=import-outside-toplevel
    converter = ctranslate2.converters.OpenNMTTFConverter(
        src_vocab=model.features_inputter.vocabulary_file,
        tgt_vocab=model.labels_inputter.vocabulary_file,
        variables=variables)
    converter.convert(export_dir, model_spec, quantization=self._quantization, force=True)
