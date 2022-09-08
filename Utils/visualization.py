import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
import seaborn           as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc

#sns.set_theme()
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
        if samples_key != 'Sample_ids':
            samples[idx] = ids[0]+'-'+samples[idx]
            class_.append(ids[0])
        
        else:
            samples[idx] = samples[idx]
            class_.append(ids[0])


    plot_data = pd.DataFrame({"Props"    : pk_props,
                              "x"        : samples,
                              "Class"    : class_
                            })

    plt.figure(figsize=(13,4))
    g = sns.barplot(x="x", y="Props", data=plot_data, palette=['limegreen', "mediumpurple"], capsize=.2, hue='Class', dodge=False)
    plt.title(title)
    g.axhline(0.5, color='r')
    if samples_key != 'Sample_ids':
        plt.xticks(rotation=90)
    plt.savefig('Images/{}.pdf'.format(title))

def mean_fusion(result1, result2, title, samples_key):
    data1 = pd.read_csv(result1)
    data2 = pd.read_csv(result2)

    pk_props = np.concatenate((data1['PK_props'].values.reshape(len(data1['PK_props'].values), 1), data2['PK_props'].values.reshape(len(data2['PK_props']),1)), axis=1)
    pk_props = np.mean(pk_props, axis=1) 

    C_props = np.concatenate((data1['C_props'].values.reshape(len(data1['C_props'].values), 1), data2['C_props'].values.reshape(len(data2['C_props']),1)), axis=1)
    C_props = np.mean(C_props, axis=1)    

    data1['PK_props'] = pk_props
    data1['C_props'] = C_props

    values = data1.values[:,1:]

    Y_true, Y_pred = values[:,0].astype(np.int64), values[:,1].astype(np.int64)
    PK_props       = values[:,2].astype(np.float64)


    acc = accuracy_score(Y_true, Y_pred)
    prf = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0.0, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_true, PK_props, pos_label=1)
    auc_metric = auc(fpr, tpr)
    
    print("==========================================================================================")
    print("Precision:{:.4f}, Recall:{:.4f}, F1-score:{:.4f}, Accuraci:{:.4f}, AUC:{:.4f}".format(prf[0][1], prf[1][1], prf[2][1], acc, auc_metric))
    print("==========================================================================================")

    data        = data1
    samples     = list(data[samples_key].values)
    patient_ids = list(data['Sample_ids'])
    pk_props    = list(data['PK_props'].values)

    class_ = []

    for idx, ids in enumerate(patient_ids):
        if samples_key != 'Sample_ids':
            samples[idx] = ids[0]+'-'+samples[idx]
            class_.append(ids[0])
        
        else:
            samples[idx] = samples[idx]
            class_.append(ids[0])


    plot_data = pd.DataFrame({"Props"    : pk_props,
                              "x"        : samples,
                              "Class"    : class_
                            })

    plt.figure(figsize=(13,4))
    g = sns.barplot(x="x", y="Props", data=plot_data, palette=['limegreen', "mediumpurple"], capsize=.2, hue='Class', dodge=False)
    plt.title(title)
    g.axhline(0.5, color='r')
    if samples_key != 'Sample_ids':
        plt.xticks(rotation=90)
    plt.savefig('Images/{}.pdf'.format(title))


def generate_final_visualization(result1, result2, title, key):

    data1 = pd.read_csv(result1)
    data2 = pd.read_csv(result2)

    sample_ids  = data1['Sample_ids'].values
    exercises   = data1['Exercise_g'].values
    repetitions = data1['Repetition'].values

    audio_pk_props = data1['PK_props'].values
    video_pk_props = data2['PK_props'].values

    pk_props = np.concatenate((data1['PK_props'].values.reshape(len(data1['PK_props'].values), 1), data2['PK_props'].values.reshape(len(data2['PK_props']),1)), axis=1)
    fusion_pk_props = np.mean(pk_props, axis=1) 

    class_ = []
    for idx, ids in enumerate(sample_ids):
            class_.append(ids[0])

    data = pd.DataFrame({'Patient IDS'           : sample_ids,
                         'Exercises'           : exercises,
                         'Repetition'           : repetitions,
                         'class'                : class_,
                         'Audio probabilities'  : audio_pk_props,
                         'Video probabilities'  : video_pk_props,
                         'Fusion probabilities' : fusion_pk_props})

    f, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 10))

    sns.stripplot(x = key, y = 'Audio probabilities', data=data, marker="o", alpha=0.3, color="blue", ax=axes[0])
    g = sns.boxplot(y = 'Audio probabilities', x = key, data = data, hue='class', palette=['limegreen', "mediumpurple"], dodge=False, flierprops={"marker": "x"}, ax=axes[0])
    g.axhline(0.5, color='r')
    g.grid(0.2)

    sns.stripplot(x = key, y = 'Video probabilities', data=data, marker="o", alpha=0.3, color="blue", ax=axes[1])
    g = sns.boxplot(y = 'Video probabilities', x = key, data = data, hue='class', palette=['limegreen', "mediumpurple"], dodge=False, flierprops={"marker": "x"}, ax=axes[1])
    g.axhline(0.5, color='r')
    g.grid(0.2)

    sns.stripplot(x = key, y = 'Fusion probabilities', data=data, marker="o", alpha=0.3, color="blue", ax=axes[2])
    g = sns.boxplot(y = 'Fusion probabilities', x = key, data = data, hue='class', palette=['limegreen', "mediumpurple"], dodge=False,flierprops={"marker": "x"}, ax=axes[2])
    g.axhline(0.5, color='r')
    g.grid(0.2)
    plt.xticks(rotation=90)

    plt.savefig('Images/{}.pdf'.format(title))

    
