"""Hypotheses file scoring."""
import pyter

def ter(ref_path, hyp_path):
  """ Compute Translation Edit Rate between two files """
  with open(ref_path) as ref_fp, open(hyp_path) as hyp_fp:
    ref_line = ref_fp.readline()
    hyp_line = hyp_fp.readline()
    ter_score = 0.0
    line_cpt = 0.0
    while ref_line and hyp_line:
      ter_score = ter_score+(pyter.ter(hyp_line.strip().split(), \
                  ref_line.strip().split()))
      line_cpt = line_cpt+1
      ref_line = ref_fp.readline()
      hyp_line = hyp_fp.readline()
  mean_ter = 1.0
  if line_cpt > 0:
    mean_ter = ter_score/line_cpt
  return mean_ter
