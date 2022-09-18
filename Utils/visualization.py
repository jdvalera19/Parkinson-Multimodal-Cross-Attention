import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
import seaborn           as sns

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn           import metrics
from sklearn.metrics   import accuracy_score
from sklearn.metrics   import precision_recall_fscore_support
from sklearn.metrics   import roc_curve, auc

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
    print("Precision:{:.4f}, Recall:{:.4f}, F1-score:{:.4f}, Accuracy:{:.4f}, AUC:{:.4f}".format(prf[0][1], prf[1][1], prf[2][1], acc, auc_metric))
    print("==========================================================================================")

def generate_final_visualization(resultAudio, resultVideo, title, key):

    dataAudio = pd.read_csv(resultAudio)
    dataVideo = pd.read_csv(resultVideo)

    sample_ids  = dataAudio['Sample_ids'].values
    exercises   = dataAudio['Exercise_g'].values
    repetitions = dataAudio['Repetition'].values

    audio_pk_props = dataAudio['PK_props'].values
    video_pk_props = dataVideo['PK_props'].values

    pk_props = np.concatenate((dataAudio['PK_props'].values.reshape(len(dataAudio['PK_props'].values), 1), dataVideo['PK_props'].values.reshape(len(dataVideo['PK_props']),1)), axis=1)
    fusion_pk_props = np.mean(pk_props, axis=1) 

    class_ = []
    for idx, ids in enumerate(sample_ids):
            class_.append(ids[0])

    data = pd.DataFrame({'Patient IDS'          : sample_ids,
                         'Exercises'            : exercises,
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

def generate_confusion_matix(results, key, modality, aucs):

    data = pd.read_csv(results)
    data = data[data['Exercise_g']==key]

    Y_true = data['Y_true']
    Y_pred = data['Y_pred']
    PK_props = data['PK_props']

    fpr, tpr, thresholds = roc_curve(Y_true, PK_props, pos_label=1)
    auc_metric = auc(fpr, tpr)
    
    #print(key)
    #print("Precision:{:.4f}, Recall:{:.4f}, F1-score:{:.4f}, Accuracy:{:.4f}, AUC:{:.4f}".format(prf[0][1], prf[1][1], prf[2][1], acc, auc_metric))
    #print("==========================================================================================")

    aucs[key] = auc_metric

    #colorlist=['white', 'mediumpurple']
    #newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)

    #confusion_matrix = metrics.confusion_matrix(Y_true, Y_pred)

    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Control', 'Parkinson'])

    #cm_display.plot(cmap = newcmp)
    #plt.title(key)
    #plt.savefig('Images/confussion matrix: {}-{}.pdf'.format(key, modality))


    




    
