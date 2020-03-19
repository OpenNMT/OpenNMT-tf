"""Hypotheses file scoring for Precision Recall and F-Measure."""


""" Compute Precision Recall and F-Measure between two files """
def fmeasure(ref_path, hyp_path):
    ref=open(ref_path)
    hyp=open(hyp_path)
    listtags=[]
    linecpt=0
    classref=[];
    classrandom=[];
    classhyp=[];
    nbrtagref={};
    nbrtaghyp={};
    nbrtagok={};
    for tag in listtags:
        nbrtagref[tag]=0
        nbrtaghyp[tag]=0
        nbrtagok[tag]=0
    for line in ref:
        line=line.strip()
        tabline=line.split(' ')
        tagcpt=0
        lineref = []
        for tag in tabline:
            lineref.append(tag)
            #classref[linecpt][tagcpt]=
            if tag in nbrtagref.keys():
                nbrtagref[tag]=nbrtagref[tag]+1
            else:
                nbrtagref[tag]=1
            tagcpt=tagcpt+1
        classref.append(lineref)
        linecpt=linecpt+1
    linemax=linecpt
    linecpt=0
    for line in hyp:
        line=line.strip()
        tabline=line.split(' ')
        tagcpt=0
        linehyp = []
        linerandom = []
        for tag in tabline:
            linehyp.append(tag)
            if tag not in listtags:
                listtags.append(tag)
            linerandom.append(tag)
            if tag == classref[linecpt][tagcpt]:
                if tag in nbrtagok.keys():
                    nbrtagok[tag]=nbrtagok[tag]+1
                else:
                    nbrtagok[tag]=1
            tagcpt=tagcpt+1
            if tag in nbrtaghyp.keys():
                nbrtaghyp[tag]=nbrtaghyp[tag]+1
            else:
                nbrtaghyp[tag]=1
        classhyp.append(linehyp)
        classrandom.append(linerandom)
        linecpt=linecpt+1

    tagcpt=0
    fullprecision=0
    fullrecall=0
    precision={}
    recall={}
    for tag in listtags:
        if tag not in nbrtagok:
            nbrtagok[tag]=0
        if tag not in nbrtaghyp:
            nbrtaghyp[tag]=0
        if tag not in nbrtagref:
            nbrtagref[tag]=0
        if nbrtaghyp[tag] != 0:
            precision[tag]=nbrtagok[tag]/nbrtaghyp[tag]
        else:
            precision[tag]=0
        if nbrtagref[tag] != 0:
            recall[tag]=nbrtagok[tag]/nbrtagref[tag]
        else:
            recall[tag]=0
        fullprecision=fullprecision+precision[tag]
        fullrecall=fullrecall+recall[tag]
        tagcpt=tagcpt+1
#        print(tag+":\t"+str(round(100*precision[tag],2))+"\t"+str(round(100*recall[tag],2))+"\t"+str(round((200*precision[tag]*recall[tag])/(precision[tag]+recall[tag]),2)))
    #    print("Recall    "+tag+": "+str(100*recall[tag]))
    fullprecision=round(100*fullprecision/tagcpt,2)/100
    fullrecall=round(100*fullrecall/tagcpt,2)/100
    fullfmeasure=(round((200*fullprecision*fullrecall)/(fullprecision+fullrecall),2))/100
    return fullprecision, fullrecall, fullfmeasure
