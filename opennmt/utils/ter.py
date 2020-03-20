"""Hypotheses file scoring."""
import numpy
import pyter

""" Compute Translation Edit Rate between two files """
def ter(ref_path, hyp_path):
    ref_fp=open(ref_path)
    hyp_fp=open(hyp_path)
    ref_line=ref_fp.readline()
    hyp_line=hyp_fp.readline()
    ter_score=0.0
    line_cpt=0.0
    while ref_line and hyp_line:
        ter_score=ter_score+(pyter.ter(hyp_line.strip().split(), ref_line.strip().split()))
        line_cpt=line_cpt+1
        ref_line=ref_fp.readline()
        hyp_line=hyp_fp.readline()
    if line_cpt > 0:
        return (ter_score/line_cpt)
    return 1.0
