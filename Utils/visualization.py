import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
import seaborn           as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc

sns.set_theme()
#sns.set(font="Arial")


def view_results(data_name):
    results = pd.read_csv(data_name)
    values = results.values[:,1:]

    Y_true, Y_pred = values[:,0].astype(np.int64), values[:,1].astype(np.int64)
    PK_props       = values[:,2].astype(np.float64)


    acc = accuracy_score(Y_true, Y_pred)
    prf = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0.0, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_true, PK_props, pos_label=1)
    auc_metric = auc(fpr, tpr)
    
    print("==========================================================================================")
    print("Precision:{:.4f}, Recall:{:.4f}, F1-score:{:.4f}, Accuraci:{:.4f}, AUC:{:.4f}".format(prf[0][1], prf[1][1], prf[2][1], acc, auc_metric))
    print("==========================================================================================")

def plot_results(result_path, title, samples_key):
    data        = pd.read_csv(result_path)
    samples     = list(data[samples_key].values)
    patient_ids = list(data['Sample_ids'])
    pk_props    = list(data['PK_props'].values)

    class_ = []

    for idx, ids in enumerate(patient_ids):
            samples[idx] = ids[0]+'-'+samples[idx]
            class_.append(ids[0])


    plot_data = pd.DataFrame({"Props"    : pk_props,
                              "x" : samples,
                              "Class"    : class_
                            })

    plt.figure(figsize=(13,4))
    g = sns.barplot(x="x", y="Props", data=plot_data, palette=['limegreen', "mediumpurple"], capsize=.2, hue='Class', dodge=False)
    plt.title(title)
    g.axhline(0.5, color='r')
    plt.xticks(rotation=60)
    plt.savefig('Images/{}.pdf'.format(title))