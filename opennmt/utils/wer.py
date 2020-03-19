"""Hypotheses file scoring."""


""" Compute Word Error Rate between two files """
def wer(ref_path, hyp_path):
    ref_fp=open(ref_path)
    hyp_fp=open(hyp_path)
    ref_line=ref_fp.readline()
    hyp_line=hyp_fp.readline()
    wer_score=0.0
    line_cpt=0.0
    while ref_line and hyp_line:
        wer_score=wer_score+sentence_wer(ref_line.strip().split(), hyp_line.strip().split())
        line_cpt=line_cpt+1
        ref_line=ref_fp.readline()
        hyp_line=hyp_fp.readline()
    if line_cpt > 0:
        return (wer_score/line_cpt)
    else return 1.0

""" Compute Word Error Rate between two sentences (as list of words) """
def sentence_wer(ref_sent, hyp_sent):
    mwer = numpy.zeros((len(ref_sent)+1)*(len(hyp_sent)+1), dtype=numpy.uint8).reshape((len(ref_sent)+1, len(hyp_sent)+1))
    for i in range(len(ref_sent)+1):
        mwer[i][0] = i
    for j in range(len(hyp_sent)+1):
        mwer[0][j] = j
    for i in range(1, len(ref_sent)+1):
        for j in range(1, len(hyp_sent)+1):
            if ref_sent[i-1] == hyp_sent[j-1]:
                mwer[i][j] = mwer[i-1][j-1]
            else:
                substitute = mwer[i-1][j-1] + 1
                insert = mwer[i][j-1] + 1
                delete = mwer[i-1][j] + 1
                mwer[i][j] = min(substitute, insert, delete)
    return (mwer[len(ref_sent)][len(hyp_sent)]/len(ref_sent))
