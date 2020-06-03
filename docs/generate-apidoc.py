import inspect
import numbers
import os
import sys
import six

import opennmt

def document_class(output_dir, class_path, base_path=None):
  with open(os.path.join(output_dir, "%s.rst" % class_path), "w") as doc:
    doc.write("%s\n" % class_path)
    doc.write("=" * len(class_path))
    doc.write("\n\n")
    doc.write(".. autoclass:: %s\n" % class_path)
    doc.write("    :members:\n")
    doc.write("    :undoc-members:\n")
    if base_path:
      doc.write("\n    Inherits from: :class:`%s`\n" % base_path)

def document_function(output_dir, function_path):
  with open(os.path.join(output_dir, "%s.rst" % function_path), "w") as doc:
    doc.write("%s\n" % function_path)
    doc.write("=" * len(function_path))
    doc.write("\n\n")
    doc.write(".. autofunction:: %s\n" % function_path)

def module_is_public(module):
  return (module.__name__.startswith("opennmt")
          and hasattr(module, "__file__")
          and module.__file__.endswith("__init__.py"))

def get_module_map(module, module_path):
  """Map true modules to exported name"""
  if not module_is_public(module):
    return {}
  m = {}
  for symbol_name in dir(module):
    if symbol_name.startswith("_"):
      continue
    symbol = getattr(module, symbol_name)
    symbol_path = "%s.%s" % (module_path, symbol_name)
    m[symbol] = symbol_path
    if inspect.ismodule(symbol):
      m.update(get_module_map(symbol, symbol_path))
  return m

def document_module(module, module_path, module_map, output_dir):
  if not module_is_public(module):
    return False
  submodules = []
  classes = []
  functions = []
  constants = []
  for symbol_name in dir(module):
    if symbol_name.startswith("_"):
      continue
    symbol = getattr(module, symbol_name)
    symbol_path = "%s.%s" % (module_path, symbol_name)
    if inspect.isclass(symbol):
      classes.append((symbol, symbol_path))
    elif inspect.isfunction(symbol) or inspect.ismethod(symbol):
      functions.append(symbol_path)
    elif inspect.ismodule(symbol):
      submodules.append((symbol_path, symbol))
    elif isinstance(symbol, (numbers.Number, six.string_types)):
      constants.append(symbol_path)

  with open(os.path.join(output_dir, "%s.rst" % module_path), "w") as doc:
    doc.write("%s module\n" % module_path)
    doc.write("=" * (len(module_path) + 7))
    doc.write("\n\n")
    doc.write(".. automodule:: %s\n\n" % module_path)

    if submodules:
      submodules = list(filter(
        lambda x: document_module(x[1], x[0], module_map, output_dir), submodules))
      if submodules:
        doc.write("Submodules\n")
        doc.write("----------\n\n")
        doc.write(".. toctree::\n\n")
        for module_path, module in submodules:
          doc.write("   %s\n" % module_path)
        doc.write("\n")

    if classes:
      doc.write("Classes\n")
      doc.write("-------\n\n")
      doc.write(".. toctree::\n\n")
      for cls, class_path in classes:
        base = cls.__bases__[0]
        while base.__name__.startswith("_"):  # Skip private parent classes.
          base = base.__bases__[0]
        if base is not object and base.__bases__[0] is tuple:  # For namedtuples.
          base = tuple
        base_path = module_map.get(base, "%s.%s" % (base.__module__, base.__name__))
        doc.write("   %s\n" % class_path)
        document_class(output_dir, class_path, base_path=base_path)

    if functions:
      doc.write("Functions\n")
      doc.write("---------\n\n")
      doc.write(".. toctree::\n\n")
      for function_path in functions:
        doc.write("   %s\n" % function_path)
        document_function(output_dir, function_path)

    if constants:
      doc.write("Constants\n")
      doc.write("---------\n\n")
      for constant_path in constants:
        doc.write("* %s\n" % constant_path)

    return True

output_dir = sys.argv[1]
os.makedirs(output_dir)
module_map = get_module_map(opennmt, "opennmt")
document_module(opennmt, "opennmt", module_map, output_dir)
