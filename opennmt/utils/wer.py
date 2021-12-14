"""Hypotheses file scoring."""


def wer(ref_path, hyp_path):
    """Compute Word Error Rate between two files"""
    with open(ref_path) as ref_fp, open(hyp_path) as hyp_fp:
        wer_score = 0.0
        line_cpt = 0.0
        for ref_line, hyp_line in zip(ref_fp, hyp_fp):
            wer_score += sentence_wer(
                ref_line.strip().split(), hyp_line.strip().split()
            )
            line_cpt = line_cpt + 1
    mean_wer = 1.0
    if line_cpt > 0:
        mean_wer = wer_score / line_cpt
    return mean_wer


def sentence_wer(ref_sent, hyp_sent):
    """Compute Word Error Rate between two sentences (as list of words)"""
    rows = len(ref_sent) + 1
    cols = len(hyp_sent) + 1
    mwer = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        mwer[i][0] = i
    for j in range(cols):
        mwer[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if ref_sent[i - 1] == hyp_sent[j - 1]:
                mwer[i][j] = mwer[i - 1][j - 1]
            else:
                substitute = mwer[i - 1][j - 1] + 1
                insert = mwer[i][j - 1] + 1
                delete = mwer[i - 1][j] + 1
                mwer[i][j] = min(substitute, insert, delete)
    sent_wer = 1.0
    if len(ref_sent) > 0:
        sent_wer = mwer[len(ref_sent)][len(hyp_sent)] / len(ref_sent)
    return sent_wer
