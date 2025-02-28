import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score


def confusion2prf(confusion):
    tp = 1.0 * np.sum([confusion[i][i] for i in range(3)])
    if tp == 0.:
        return 0., 0., 0.

    prec = tp / (np.sum(confusion[:4,:3]))
    rec = tp / (np.sum(confusion[:3,:4]))
    f1 = 2.0 / (1.0 / prec + 1.0 / rec)
    return prec,rec,f1


def evaluation(all_golds, all_preds):
    acc = accuracy_score(all_golds, all_preds)
    confu = confusion_matrix(all_golds, all_preds)
    cl_report = classification_report(all_golds, all_preds, digits=4)
    prec, rec, f1 = confusion2prf(confu)
    micro_f1 = f1_score(all_golds, all_preds, average='micro')
    report = "Prec=%.4f, Rec=%.4f, F1=%.4f, Acc=%.4f, MICRO_F1=%.4f" % (prec, rec, f1, acc, micro_f1)

    print(cl_report)
    print(confu, flush=True)
    print(report)
    return f1
