"""ARK data file to TFRecords converter.

The scripts takes the ARK data file and optionally the indexed target text
to write aligned source and target data.
"""

from __future__ import print_function

import argparse
import io
import numpy as np
import tensorflow as tf

from opennmt.inputters.record_inputter import write_sequence_record


def consume_next_vector(ark_file, dtype):
  """Consumes the next vector.

  Args:
    ark_file: The ARK data file.

  Returns:
    The next vector as a 2D Numpy array.
  """
  idx = None
  vector = []

  for line in ark_file:
    line = line.strip()
    fields = line.split()

    if not idx:
      idx = fields[0]
      fields.pop(0)
      fields.pop(0)

    end = fields and fields[-1] == "]"

    if end:
      fields.pop()

    if fields:
      vector.append(fields)

    if end:
      break

  return idx, np.asarray(vector, dtype=dtype)

def consume_next_text(text_file):
  """Consumes the next text line from `text_file`."""
  idx = None
  text = text_file.readline()

  if text:
    tokens = text.strip().split()
    idx = tokens[0]
    tokens.pop(0)
    text = " ".join(tokens)

  return idx, text

def write_text(text, writer):
  """Serializes a line of text."""
  writer.write(text)
  writer.write("\n")

def ark_to_records_aligned(ark_filename, text_filename, out_prefix, dtype=np.float32):
  """Converts ARK and text datasets to aligned TFRecords and text datasets."""
  record_writer = tf.python_io.TFRecordWriter(out_prefix + ".records")
  text_writer = io.open(out_prefix + ".txt", encoding="utf-8", mode="w")

  ark_buffer = {}
  text_buffer = {}
  count = 0

  def _write_example(vector, text):
    write_sequence_record(vector, record_writer)
    write_text(text, text_writer)

  def _search_aligned():
    for idx in ark_buffer:
      if idx in text_buffer:
        vector = ark_buffer[idx]
        text = text_buffer[idx]

        del ark_buffer[idx]
        del text_buffer[idx]

        return vector, text

    return None, None

  with io.open(ark_filename, encoding="utf-8") as ark_file, open(text_filename, encoding="utf-8") as text_file: #pylint: disable=line-too-long
    while True:
      ark_idx, vector = consume_next_vector(ark_file, dtype=dtype)
      text_idx, text = consume_next_text(text_file)

      if not ark_idx and not text_idx:
        # Both files are empty.
        break

      if ark_idx == text_idx:
        # If the indices match, write the example.
        _write_example(vector, text)
        count += 1
      else:
        # Otherwise store the entries.
        if ark_idx:
          ark_buffer[ark_idx] = vector
        if text_idx:
          text_buffer[text_idx] = text

        # Look if we can now find aligned entries.
        vector, text = _search_aligned()

        if vector is not None:
          _write_example(vector, text)
          count += 1

  # Search alignments in stored entries.
  while True:
    vector, text = _search_aligned()
    if vector is None:
      break
    _write_example(vector, text)
    count += 1

  record_writer.close()
  text_writer.close()

  print("Saved {} aligned records.".format(count))

def ark_to_records(ark_filename, out_prefix, dtype=np.float32):
  """Converts ARK dataset to TFRecords."""
  record_writer = tf.python_io.TFRecordWriter(out_prefix + ".records")
  count = 0

  with io.open(ark_filename, encoding="utf-8") as ark_file:
    while True:
      ark_idx, vector = consume_next_vector(ark_file, dtype=dtype)
      if not ark_idx:
        break
      write_sequence_record(vector, record_writer)
      count += 1

  record_writer.close()
  print("Saved {} records.".format(count))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ark", required=True,
                      help="Indexed ARK data file.")
  parser.add_argument("--txt",
                      help=("Indexed target text data file "
                            "(must set it to align source and target files)."))
  parser.add_argument("--out", required=True,
                      help="Output files prefix (will be suffixed by .records and .txt).")
  parser.add_argument("--dtype", default="float32",
                      help="Vector dtype")
  args = parser.parse_args()
  dtype = np.dtype(args.dtype)

  if args.txt:
    ark_to_records_aligned(args.ark, args.txt, args.out, dtype=dtype)
  else:
    ark_to_records(args.ark, args.out, dtype=dtype)

if __name__ == "__main__":
  main()
