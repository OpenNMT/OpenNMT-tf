"""Various utility functions to use throughout the project."""


def count_lines(filename):
  """Returns the number of lines of the file `filename`."""
  with open(filename) as f:
    i = 0
    for i, _ in enumerate(f):
      pass
    return i + 1
