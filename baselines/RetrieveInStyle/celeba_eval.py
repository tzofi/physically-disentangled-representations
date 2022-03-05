import sys
import pdb
import numpy as np
import collections
from scipy.cluster import hierarchy as sphac
from scipy.spatial import distance as spdist
import numpy as np
import warnings
from sklearn import cluster
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_fscore_support
from finch import FINCH
from scipy.stats import mode

def weighted_purity(Y, C):
    """Computes weighted purity of HAC at one particular clustering "C".
    Y, C: np.array([...]) containing unique cluster indices (need not be same!)
    Note: purity --> 1 as the number of clusters increase, so don't look at this number alone!
    """

    purity = 0.
    uniq_clid, clustering_skew = np.unique(C, return_counts=True)
    num_samples = np.zeros(uniq_clid.shape)
    # loop over all predicted clusters in C, and measure each one's cardinality and purity
    for k in uniq_clid:
        # gt labels for samples in this cluster
        k_gt = Y[np.where(C == k)[0]]
        values, counts = np.unique(k_gt, return_counts=True)
        # technically purity = max(counts) / sum(counts), but in WCP, the sum(counts) multiplies to "weight" the clusters
        purity += max(counts)

    purity /= Y.shape[0]
    return purity, clustering_skew


def WCP(Y_pred, Y):
    gt_num_clusters = len(set(Y_pred.flatten()))

    predicted_label = np.zeros((len(Y_pred), 1)).astype('int')

    wcp = 0
    for i in range(gt_num_clusters):
        idx = np.where(Y_pred == i)
        actual_cluster_i = Y[idx]

        id = mode(actual_cluster_i)[0][0]
        predicted_label[idx,:]= id
        wcp = wcp + len(np.where(actual_cluster_i==id)[0])

    wcp = wcp/len(Y_pred)

    return wcp, predicted_label


def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    #from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind


def NMI(Y, C):
    """Normalized Mutual Information: Clustering performance between ground-truth Y and prediction C
    Based on https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf

    Result matches examples on pdf
    Example:
    Y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    C = np.array([1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2])
    NMI(Y, C) = 0.1089

    C = np.array([1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    NMI(Y, C) = 0.2533
    """

    def entropy(labels):
        # H(Y) and H(C)
        H = 0.
        for k in np.unique(labels):
            p = (labels == k).sum() / labels.size
            H -= p * np.log2(p)
        return H

    def h_y_given_c(labels, pred):
        # H(Y | C)
        H = 0.
        for c in np.unique(pred):
            p_c = (pred == c).sum() / pred.size
            labels_c = labels[pred == c]
            for k in np.unique(labels_c):
                p = (labels_c == k).sum() / labels_c.size
                H -= p_c * p * np.log2(p)
        return H

    h_Y = entropy(Y)
    h_C = entropy(C)
    h_Y_C = h_y_given_c(Y, C)
    # I(Y; C) = H(Y) - H(Y|C)
    mi = h_Y - h_Y_C
    # NMI = 2 * MI / (H(Y) + H(C))
    nmi = 2 * mi / (h_Y + h_C)
    return nmi

def hac_cluster(data,req_n_clusters):
    algorithm = cluster.AgglomerativeClustering(n_clusters=req_n_clusters, linkage='ward')
    warnings.filterwarnings("ignore")
    algorithm.fit(data)

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_
    else:
        y_pred = algorithm.predict(data)
    return y_pred

"""
## Data
#data_vae = np.load("vae_feats.npz", allow_pickle=True)
#data_vae = np.load("vae_utkface_race_feats_id.npz", allow_pickle=True)
data_vae = np.load("vae_bbt_feats.npz", allow_pickle=True)
#data_vae = np.load("vae_bbt_single_image_feats.npz", allow_pickle=True)
#data_vae = np.load("vae_rafdb_feats.npz", allow_pickle=True)
d_vae, gt_vlabels = data_vae["feats"], data_vae["labels"]#[:,:,19]
#data_ir = np.load("inverse_rendered_feats.npz", allow_pickle=True)
#data_ir = np.load("inverse_rendered_utkrace_cycle_light_feats_id.npz", allow_pickle=True)
#data_ir = np.load("inverse_rendered_utkrace_cycle_light_feats_id.npz", allow_pickle=True)
#data_ir = np.load("inverse_rendered_bbt_cycle_light_feats.npz", allow_pickle=True)
#data_ir = np.load("inverse_rendered_bbt_cycle_light_single_image_feats.npz", allow_pickle=True)
#data_ir = np.load("inverse_rendered_buffy_cycle_light_feats.npz", allow_pickle=True)
data_ir = np.load("inverse_rendered_bbt_cycle_light_feats.npz", allow_pickle=True)
#data_ir = np.load("inverse_rendered_ck_cycle_light_feats.npz", allow_pickle=True)
#data_ir = np.load("inverse_rendered_rafdb_cycle_light_feats.npz", allow_pickle=True)
#data_ir = np.load("inverse_rendered_rafdb_feats.npz", allow_pickle=True)
d_ir, gt_irlabels = data_ir["feats"][:,:,:], data_ir["labels"]#[:,:,19]
print(d_ir.shape)
print(np.unique(gt_irlabels))
"""

files = [sys.argv[1]]
for fname in files:
    data = np.load(fname, allow_pickle=True)
    d, gt = data["feats"], data["labels"]
    accs = []
    nums = []
    for label in range(gt.shape[-1]):
        gt_i = gt[:,label]
        y_pred = hac_cluster(d, np.unique(gt_i).shape[0])
        gt_i = gt_i.flatten()

        print("Results for {}, {}:".format(fname, d.shape))
        nmi = NMI(gt_i, y_pred)
        wcp, predicted = WCP(y_pred,gt_i)
        prec = precision_score(gt_i, predicted, average='weighted')
        rec = recall_score(gt_i, predicted, average='weighted')
        f1 = f1_score(gt_i, predicted, average='weighted')
        #acc, _ = cluster_acc(predicted, gt)
        #print(acc)

        if accs == []:
            accs = np.array([nmi, prec, rec, f1])
        else:
            accs += np.array([nmi, prec, rec, f1])

        print('NMI: {}, WCP: {}, Precision: {}, Recall: {}, F1: {}, No. of Clusters: {}'.format(nmi,wcp,prec,rec,f1,np.unique(gt_i).shape[0]))

    print(accs/gt.shape[-1])

    exit()

## HAC
y_pred_vae = hac_cluster(np.squeeze(d_vae, axis=1), np.unique(gt_vlabels).shape[0])
y_pred_ir = hac_cluster(np.squeeze(d_ir, axis=1), np.unique(gt_irlabels).shape[0])

## FINCH
#c_vae, n_c_vae, req_c_vae = FINCH(np.squeeze(d_vae, axis=1))
#c_ir, n_c_ir, req_c_ir = FINCH(np.squeeze(d_ir, axis=1))

gt_vlabels = gt_vlabels.flatten()
gt_irlabels = gt_irlabels.flatten()

### HAC

# metrics -- VAE
nmi = NMI(gt_vlabels, y_pred_vae)
#wcp = weighted_purity(gt_vlabels, y_pred_vae)[0]
wcp, vpredicted = WCP(y_pred_vae,gt_vlabels)
prec = precision_score(gt_vlabels, vpredicted, average='weighted')
rec = recall_score(gt_vlabels, vpredicted, average='weighted')
f1 = f1_score(gt_vlabels, vpredicted, average='weighted')
# print, store and return
print('VAE. NMI: {}, WCP: {}, Precision: {}, Recall: {}, F1: {}, No. of Clusters: {}'.format(nmi,wcp,wcp_new,np.unique(gt_vlabels).shape[0]))

# metrics -- Inverse Rendering
nmi = NMI(gt_irlabels, y_pred_ir)
wcp = weighted_purity(gt_irlabels, y_pred_ir)[0]
# print, store and return
print('IR. NMI: {}, WCP: {}, No. of Clusters: {}'.format(nmi,wcp,np.unique(gt_irlabels).shape[0]))

### FINCH

level_ = 1


"""
# metrics -- VAE
nmi = NMI(gt_vlabels, c_vae[:,level_])
wcp = weighted_purity(gt_vlabels, c_vae[:,level_])[0]
# print, store and return
print('VAE. NMI: {}, WCP: {}, No. of Clusters: {}'.format(nmi,wcp,np.unique(c_vae[:,level_]).shape[0]))

# metrics -- Inverse Rendering
nmi = NMI(gt_irlabels, c_ir[:,level_])
wcp = weighted_purity(gt_irlabels, c_ir[:,level_])[0]
# print, store and return
print('IR. NMI: {}, WCP: {}, No. of Clusters: {}'.format(nmi,wcp,np.unique(c_ir[:,level_]).shape[0]))
"""
#c_, nc_, req_c = FINCH(np.squeeze(d_ir, axis=1), initial_rank=None, req_clust=np.unique(c_vae[:,level_]).shape[0], distance='cosine', verbose=True)

# metrics -- Inverse Rendering
#nmi = NMI(gt_irlabels, req_c)
#wcp = weighted_purity(gt_irlabels, req_c)[0]
## print, store and return
#print('IR. NMI: {}, WCP: {}, No. of Clusters: {}'.format(nmi,wcp,np.unique(req_c).shape[0]))
