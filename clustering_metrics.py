import numpy as np
import math
from scipy.optimize import linear_sum_assignment as linear_assignment


def calc_ACC(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scipy installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def calc_NMI(A,B):
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    # MIhat = 2.0*MI/(Hx+Hy)
    MIhat = MI / max(Hx, Hy)
    return MIhat

def jaccard(cluster, label):
    dist_cluster = np.abs(np.tile(cluster, (len(cluster), 1)) -
                          np.tile(cluster, (len(cluster), 1)).T)
    dist_label = np.abs(np.tile(label, (len(label), 1)) -
                        np.tile(label, (len(label), 1)).T)
    a_loc = np.argwhere(dist_cluster+dist_label == 0)
    n = len(cluster)
    a = (a_loc.shape[0]-n)/2
    same_cluster_index = np.argwhere(dist_cluster == 0)
    same_label_index = np.argwhere(dist_label == 0)
    bc = same_cluster_index.shape[0]+same_label_index.shape[0]-2*n-2*a
    return a/(a+bc)

def calc_Purity(label, cluster):
    cluster = np.array(cluster)
    label = np. array(label)
    indedata1 = {}
    for p in np.unique(label):
        indedata1[p] = np.argwhere(label == p)
    indedata2 = {}
    for q in np.unique(cluster):
        indedata2[q] = np.argwhere(cluster == q)

    count_all = []
    for i in indedata1.values():
        count = []
        for j in indedata2.values():
            a = np.intersect1d(i, j).shape[0]
            count.append(a)
        count_all.append(count)

    return sum(np.max(count_all, axis=0))/len(cluster)
