import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
import seaborn           as sns
import ast

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn           import metrics
from sklearn.metrics   import accuracy_score
from sklearn.metrics   import precision_recall_fscore_support
from sklearn.metrics   import roc_curve, auc

def view_results_2(data_name):
    results = pd.read_csv(data_name)
    values = results.values[:,1:]

    # Limpiar las cadenas y convertirlas a floats
    def clean_value(val):
        # Eliminar los corchetes y convertir la cadena a float
        return float(ast.literal_eval(val)[0]) if isinstance(val, str) else val

    # Aplicar la limpieza de valores
    Y_true = np.array([clean_value(v) for v in values[:,0]]).astype(np.float64).astype(np.int64)
    Y_pred = np.array([clean_value(v) for v in values[:,1]]).astype(np.float64).astype(np.int64)
    PK_props = np.array([clean_value(v) for v in values[:,2]]).astype(np.float64)

    # Calcular las métricas
    acc = accuracy_score(Y_true, Y_pred)
    prf = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0.0, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_true, PK_props, pos_label=1)
    auc_metric = auc(fpr, tpr)

    print("==========================================================================================")
    print("Precision:{:.4f}, Recall:{:.4f}, F1-score:{:.4f}, Accuracy:{:.4f}, AUC:{:.4f}".format(prf[0][1], prf[1][1], prf[2][1], acc, auc_metric))
    print("==========================================================================================")

def view_results(data_name, n_bootstrap=1000):
    results = pd.read_csv(data_name)
    values = results.values[:,1:]

    #Y_true, Y_pred = values[:,0].astype(np.int64), values[:,1].astype(np.int64)
    Y_true, Y_pred = values[:,0].astype(np.float64).astype(np.int64), values[:,1].astype(np.float64).astype(np.int64) #PRUEBA ALEJANDRA
    PK_props       = values[:,2].astype(np.float64)


    acc = accuracy_score(Y_true, Y_pred)
    prf = precision_recall_fscore_support(Y_true, Y_pred, zero_division=0.0, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_true, PK_props, pos_label=1)
    auc_metric = auc(fpr, tpr)
    
    # Bootstrap para calcular desviación estándar del AUC
    auc_scores = []
    for _ in range(n_bootstrap):
        # Muestreo con reemplazo
        indices = np.random.choice(len(Y_true), len(Y_true), replace=True)
        Y_true_sample = Y_true[indices]
        PK_props_sample = PK_props[indices]
        
        # Calcular AUC en el bootstrap sample
        fpr_sample, tpr_sample, _ = roc_curve(Y_true_sample, PK_props_sample, pos_label=1)
        auc_sample = auc(fpr_sample, tpr_sample)
        auc_scores.append(auc_sample)
    
    # Calcular desviación estándar del AUC a partir de los valores obtenidos en el bootstrap
    auc_std = np.std(auc_scores)

    print("==========================================================================================")
    print("Precision:{:.4f}, Recall:{:.4f}, F1-score:{:.4f}, Accuracy:{:.4f}, AUC:{:.4f} ± {:.4f}".format(prf[0][1], prf[1][1], prf[2][1], acc, auc_metric, auc_std))
    print("==========================================================================================")

def audio_visual_evaluation(resultAudio, resultVideo):

    dataAudio = pd.read_csv(resultAudio)
    dataVideo = pd.read_csv(resultVideo)

    Y_true    = dataAudio['Y_true'].values

    alpha = 0.1
    delta = 0.05

    labels = []
    aucs   = []
    while alpha <= 0.9:
        fusion_predictions = []

        audio_pk_props = dataAudio['PK_props'].values
        video_pk_props = dataVideo['PK_props'].values

        video_pk_props = video_pk_props*alpha
        audio_pk_props = audio_pk_props*round(1-alpha, 2)

        pk_props        = np.concatenate((audio_pk_props.reshape(len(audio_pk_props), 1), video_pk_props.reshape(len(video_pk_props),1)), axis=1)
        fusion_pk_props = np.sum(pk_props, axis=1)

        for idx in range(len(fusion_pk_props)):
            if fusion_pk_props[idx] <= 0.5:
                fusion_predictions.append(0)

            else:
                fusion_predictions.append(1)

        acc = accuracy_score(Y_true, fusion_predictions)
        prf = precision_recall_fscore_support(Y_true, fusion_predictions, zero_division=0.0, pos_label=1)
        fpr, tpr, thresholds = roc_curve(Y_true, fusion_pk_props, pos_label=1)
        auc_metric = auc(fpr, tpr)

        print("==========================================================================================")
        print("V:{} , A:{}, Precision:{:.4f}, Recall:{:.4f}, F1-score:{:.4f}, Accuracy:{:.4f}, AUC:{:.4f}".format(alpha, round(1-alpha, 2), prf[0][1], prf[1][1], prf[2][1], acc, auc_metric))
        print("==========================================================================================")

        labels.append(alpha)
        aucs.append(round(auc_metric, 3))

        alpha += delta
        alpha = round(alpha, 2)

    return {'Alphas': labels, 'AUC': aucs}

def plot_alpha(data_plot, title):

    f, axes = plt.subplots(figsize=(18, 5))
    g = sns.barplot(data=data_plot, x="Alphas", y="AUC", palette=['limegreen', "mediumpurple", 'gray'], hue='Exercises', axes=axes)
    g.set_ylim(0, 1)
    for container in g.containers:
        g.bar_label(container,rotation=80)

    plt.axvline(x=7.5, color="red", linestyle="--")
    plt.axvline(x=8.5, color="red", linestyle="--")
    plt.axvline(x=9.5, color="red", linestyle="--")
    plt.xticks(rotation=70)
    plt.legend(loc='lower left')
    plt.savefig('Images/{}.pdf'.format(title), bbox_inches='tight')
    


def generate_final_visualization(resultAudio, resultVideo, title, key, alpha):

    plt.rcParams["font.family"] = "DejaVu Sans"

    dataAudio = pd.read_csv(resultAudio)
    dataVideo = pd.read_csv(resultVideo)
    Y_true    = dataAudio['Y_true'].values

    sample_ids  = dataAudio['Sample_ids'].values
    exercises   = dataAudio['Exercise_g'].values
    repetitions = dataAudio['Repetition'].values

    audio_pk_props = dataAudio['PK_props'].values
    video_pk_props = dataVideo['PK_props'].values

    pk_props = np.concatenate((dataAudio['PK_props'].values.reshape(len(dataAudio['PK_props'].values), 1), dataVideo['PK_props'].values.reshape(len(dataVideo['PK_props']),1)), axis=1)
    pk_props[:, 0] = pk_props[:, 0]*alpha
    pk_props[:, 1] = pk_props[:, 1]*(1-alpha) 
    fusion_pk_props = np.sum(pk_props, axis=1) 

    fpr, tpr, thresholds = roc_curve(Y_true, fusion_pk_props, pos_label=1) 
    auc_metric = auc(fpr, tpr)

    print('---------------------------------------')
    print('AUC:{}'.format(round(auc_metric,2)))
    print('---------------------------------------')

    class_ = []
    for idx, ids in enumerate(sample_ids):
        if ids[0] == 'C':
            class_.append('Control')
        else:
            class_.append('Parkinson')

    data = pd.DataFrame({'Patient IDS'          : sample_ids,
                         'Exercises'            : exercises,
                         'Repetition'           : repetitions,
                         'Samples'                : class_,
                         'Audio probabilities'  : audio_pk_props,
                         'Video probabilities'  : video_pk_props,
                         'Fusion probabilities' : fusion_pk_props})

    f, axes = plt.subplots(3, 1, sharex=True, figsize=(20, 8))

    sns.stripplot(x = key, y = 'Audio probabilities', data=data, marker="o", alpha=0.3, color="blue", ax=axes[0])
    g = sns.boxplot(y = 'Audio probabilities', x = key, data = data, hue='Samples', palette=['limegreen', "mediumpurple"], dodge=False, flierprops={"marker": "x"}, ax=axes[0])
    g.axhline(0.5, color='r', alpha=0.7)
    g.set_xlabel('')
    g.set_ylabel('')
    g.set_title('a) Audio samples', fontsize=15)
    g.get_legend().remove()
    g.grid(0.1)

    sns.stripplot(x = key, y = 'Video probabilities', data=data, marker="o", alpha=0.3, color="blue", ax=axes[1])
    g = sns.boxplot(y = 'Video probabilities', x = key, data = data, hue='Samples', palette=['limegreen', "mediumpurple"], dodge=False, flierprops={"marker": "x"}, ax=axes[1])
    g.axhline(0.5, color='r', alpha=0.7)
    g.set_xlabel('')
    g.set_ylabel('Probabilities', fontsize=15)
    g.set_title('b) Video samples', fontsize=15)
    g.grid(0.1)

    sns.stripplot(x = key, y = 'Fusion probabilities', data=data, marker="o", alpha=0.3, color="blue", ax=axes[2])
    g = sns.boxplot(y = 'Fusion probabilities', x = key, data = data, hue='Samples', palette=['limegreen', "mediumpurple"], dodge=False,flierprops={"marker": "x"}, ax=axes[2])
    g.axhline(0.5, color='r', alpha=0.7)
    g.set_ylabel('')
    g.set_xlabel('Patient IDS', fontsize=15)
    g.set_title('c) Fusion samples', fontsize=15)
    g.get_legend().remove()
    g.grid(0.1)

    plt.xticks(rotation=0)
    plt.savefig('Images/{}.pdf'.format(title), bbox_inches='tight')

def generate_confusion_matix(results, key, modality, aucs):

    data = pd.read_csv(results)
    data = data[data['Exercise_g']==key]

    Y_true      = data['Y_true'].values
    Y_pred      = data['Y_pred'].values
    PK_props    = data['PK_props'].values
    Samples_ids = data['Sample_ids'].values
    Exercises   = data['Exercise_g'].values
    Repetitions = data['Repetition'].values

    fpr, tpr, thresholds = roc_curve(Y_true, PK_props, pos_label=1)
    auc_metric = auc(fpr, tpr)
    
    #print(key)
    #print("Precision:{:.4f}, Recall:{:.4f}, F1-score:{:.4f}, Accuracy:{:.4f}, AUC:{:.4f}".format(prf[0][1], prf[1][1], prf[2][1], acc, auc_metric))
    #print("==========================================================================================")

    aucs[key] = auc_metric

    '''colorlist=['white', 'mediumpurple']
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)

    confusion_matrix = metrics.confusion_matrix(Y_true, Y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Control', 'Parkinson'])

    cm_display.plot(cmap = newcmp)
    plt.title(key)
    plt.savefig('Images/confussion matrix: {}-{}.pdf'.format(key, modality))'''

    print(key, 'ok')
    for idx in range(len(Samples_ids[Y_true==Y_pred])):
        print(Samples_ids[Y_true==Y_pred][idx], Exercises[Y_true==Y_pred][idx], Repetitions[Y_true==Y_pred][idx])
    
    print(key, 'bad')
    for idx in range(len(Samples_ids[Y_true!=Y_pred])):
        print(Samples_ids[Y_true!=Y_pred][idx], Exercises[Y_true!=Y_pred][idx], Repetitions[Y_true!=Y_pred][idx])


    




    
