import os

import pandas            as pd
import seaborn           as sns
import numpy             as np
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n{:d}".format(pct, absolute)

def get_dataset_information(data_path):
    list_labels    = []
    list_patients  = []
    list_exercises = []
    list_samples   = []

    labels = sorted(os.listdir(data_path))

    for label in labels:
        patients_path = data_path + label + "/"
        patients = os.listdir(patients_path)
        
        for patient in patients:
            exercises_path = patients_path + patient + "/"
            exercises = os.listdir(exercises_path)
            
            for exercise in exercises:
                samples_path = exercises_path + exercise + "/"
                samples = os.listdir(samples_path)
                
                list_samples += samples
                
                temp_list_labels = [label for sample in samples]
                list_labels += temp_list_labels
                
                temp_list_patients = [patient for sample in samples]
                list_patients += temp_list_patients
                
                temp_list_exercises = [exercise for sample in samples]
                list_exercises += temp_list_exercises
                
    dataset_description = pd.DataFrame({"Samples":list_samples,
                                        "Labels":list_labels,
                                        "Patients":list_patients,
                                        "Exercises":list_exercises})

    return dataset_description

def plot_data(dataFrame, label):

    total_words  = dataFrame[dataFrame['Labels']==label][dataFrame['Exercises']=='Words'].shape[0]
    total_vowels = dataFrame[dataFrame['Labels']==label][dataFrame['Exercises']=='Vowels'].shape[0]
    total_phonemes = dataFrame[dataFrame['Labels']==label][dataFrame['Exercises']=='Phonemes'].shape[0]

    data      = np.array([total_vowels, total_phonemes, total_words])
    colors    = ['limegreen', "mediumpurple", "gray"]
    myexplode = [0, 0, 0.2]

    plt.pie(data, colors = colors, autopct= lambda pct: func(pct, data), explode = myexplode, startangle = 90)
    plt.title(label)
    plt.legend(['Vowels', 'Phonemes', 'Words'])
    plt.savefig("Images/{}_exercises_distribution.png".format(label), bbox_inches='tight', pad_inches = 0, transparent=True)
    plt.close()

if __name__== "__main__":
    data_path = '/data/Datasets/Parkinson/Facial_Expression_Dataset/AudioVisualDataset_v7/'

    dataset_dataFrame = get_dataset_information(data_path)

    plot_data(dataset_dataFrame, 'Control')
    plot_data(dataset_dataFrame, 'Parkinson')