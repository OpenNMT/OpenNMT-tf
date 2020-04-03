"""Hypotheses file scoring for Precision Recall and F-Measure."""

def fmeasure(ref_path,
             hyp_path,
             return_precision_only=False,
             return_recall_only=False,
             return_fmeasure_only=False):
  """Compute Precision Recall and F-Measure between two files"""
  with open(ref_path) as ref_fp, open(hyp_path) as hyp_fp:
    list_null_tags = ["X", "null", "NULL", "Null", "O"]
    listtags = []
    linecpt = 0
    classref = []
    classrandom = []
    classhyp = []
    nbrtagref = {}
    nbrtaghyp = {}
    nbrtagok = {}
    for tag in listtags:
      nbrtagref[tag] = 0
      nbrtaghyp[tag] = 0
      nbrtagok[tag] = 0
    for line in ref_fp:
      line = line.strip()
      tabline = line.split(' ')
      tagcpt = 0
      lineref = []
      for tag in tabline:
        lineref.append(tag)
        if tag in nbrtagref.keys() and tag not in list_null_tags:
          nbrtagref[tag] = nbrtagref[tag]+1
        else:
          nbrtagref[tag] = 1
        tagcpt = tagcpt+1
      classref.append(lineref)
      linecpt = linecpt+1
    linecpt = 0
    for line in hyp_fp:
      line = line.strip()
      tabline = line.split(' ')
      tagcpt = 0
      linehyp = []
      linerandom = []
      for tag in tabline:
        linehyp.append(tag)
        if tag not in listtags:
          listtags.append(tag)
        linerandom.append(tag)
        if tag == classref[linecpt][tagcpt]:
          if tag in nbrtagok.keys():
            nbrtagok[tag] = nbrtagok[tag]+1
          else:
            nbrtagok[tag] = 1
        tagcpt = tagcpt+1
        if tag in nbrtaghyp.keys():
          nbrtaghyp[tag] = nbrtaghyp[tag]+1
        else:
          nbrtaghyp[tag] = 1
      classhyp.append(linehyp)
      classrandom.append(linerandom)
      linecpt = linecpt+1

  tagcpt = 0
  fullprecision = 0
  fullrecall = 0
  precision = {}
  recall = {}
  fulltagok = 0.00
  fulltaghyp = 0.00
  fulltagref = 0.00
  for tag in listtags:
    if tag not in nbrtagok:
      nbrtagok[tag] = 0
    if tag not in nbrtaghyp:
      nbrtaghyp[tag] = 0
    if tag not in nbrtagref:
      nbrtagref[tag] = 0
    if nbrtaghyp[tag] != 0:
      precision[tag] = nbrtagok[tag]/nbrtaghyp[tag]
    else:
      precision[tag] = 0
    if nbrtagref[tag] != 0:
      recall[tag] = nbrtagok[tag]/nbrtagref[tag]
    else:
      recall[tag] = 0
    if tag not in list_null_tags:
      fulltagok = fulltagok+nbrtagok[tag]
      fulltaghyp = fulltaghyp+nbrtaghyp[tag]
      fulltagref = fulltagref+nbrtagref[tag]
#      fullprecision = fullprecision+precision[tag]
#      fullrecall = fullrecall+recall[tag]
    tagcpt = tagcpt+1
  fullprecision = round(100*fulltagok/fulltaghyp, 2)/100
  fullrecall = round(100*fulltagok/fulltagref, 2)/100
  fullfmeasure = (round((200*fullprecision*fullrecall)/(fullprecision+fullrecall), 2))/100
  if return_precision_only:
    return fullprecision
  if return_recall_only:
    return fullrecall
  if return_fmeasure_only:
    return fullfmeasure
  return fullprecision, fullrecall, fullfmeasure
