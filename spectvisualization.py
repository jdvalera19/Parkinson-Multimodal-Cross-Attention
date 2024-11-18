import os
import random
import numpy as np
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
import pickle

import pandas as pd

from torchvision import transforms
import torchaudio.transforms as audio_transforms
from torch.utils.data import DataLoader
from joblib import dump, load
import os

#Custom librarian
from Utils.dataset       import *
from Utils.nets          import *
from Utils.visualization import *
from Utils.model2d import *
from Utils.model3d import *
from multiattn import RFBMultiHAttnNetwork_V4

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
def set_seed(seed_value):
    #Set seed for reproducibility.
    random.seed(seed_value)    # Python random module.
    np.random.seed(seed_value) # Numpy module.
    torch.manual_seed(seed_value) # PyTorch to initialize random numbers.
    torch.cuda.manual_seed(seed_value) # CUDA randomness.
    torch.cuda.manual_seed_all(seed_value) # CUDA for all GPUs.
    torch.backends.cudnn.deterministic = True  # To make sure that every time you run your script on the same hardware, you get the same output.
    torch.backends.cudnn.benchmark = False
"""
    
if __name__== "__main__":

    # Set a seed for reproducibility
    #set_seed(42)
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #-------------------------------------------------------------------
    # Define the model parameters
    #-------------------------------------------------------------------
    lr          = 0.001
    epoch       = 70
    batch_size  = 5
    exercise    = 'Words'
    #path_data   = '/data/arumota_pupils/Jose/AudioVisualData_v7'
    path_data   = '/data/franklin_pupils/Jose/Dataset/AudioVisualData_v7'
    note        = 'PRUEBA_SANTIAGO_SELF_ATENCION_FINAL_DataAugmentation_2Cabeza0.5AtenciónFinal_atenciónEmbebidos_LR_PRUEBA_AUDIO2D_VIDEO3D_PesosGuardados:weights' 
    #note        = 'DataAugmentation_4MultiCabezaAtenciónSimpleDrop0.5_atenciónFeatures_AUDIO2D_VIDEO3D_PesosGuardados:weights' 
    s_duration  = False

    #-------------------------------------------------------------------
    # Select the GPU to improve the evaluation stage
    #-------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('-------------------------------------------------------------------')
        print("Using GPU for training:", torch.cuda.get_device_name())
        print('-------------------------------------------------------------------')
    else:
        device = torch.device("cpu")
        print('-------------------------------------------------------------------')
        print("Failed to find GPU, using CPU instead")
        print('-------------------------------------------------------------------')

    #-------------------------------------------------------------------
    # List with all patients and list to save the prediction per patient
    #-------------------------------------------------------------------
    parkinson_patients = sorted(os.listdir("{}/Parkinson".format(path_data)))
    control_patients   = sorted(os.listdir("{}/Control".format(path_data)))
    patients           = control_patients + parkinson_patients


    Y_true_g       = []
    Y_pred_g       = []
    PK_props_g     = []
    C_props_g      = []
    samples_ids_g  = []
    exercises_g    = []
    repetitions_g  = []
    val_loss_g     = []
    lr_history_g   = []

    _, _, _, _, _, duration_video = generate_train_and_test_sets(path_base   = path_data, 
                                                                 patient_val = ['P0'],
                                                                 exercise_s  = exercise, 
                                                                 duration    = s_duration)
    
    videos, labels, samples_type, exercises_s, repetitions_s = load_video_data(path_base   = path_data,
                                                                               exercise_s  = exercise, 
                                                                               duration    = duration_video)  

    # Inicializar el modelo de atención fuera del bucle de pacientes
    cross_model = FinalMultiHeadAttention(dim_input=256, dim_query=128, dim_key=128, dim_value=128, num_heads=2).to(device)
    concat_emb_model = EmbeddingConcatenation().to(device)
    optimizer_cross = torch.optim.Adam(cross_model.parameters(), lr=lr, weight_decay=0.001)
    
    for patient in patients:
        
        #audios_Train, audios_Test, videos_Train, videos_Test, duration_audio, duration_video = generate_train_and_test_sets(path_base   = path_data, 
        #                                                                                  patient_val = [patient],
        #                                                                                  exercise_s  = exercise, 
        #                                                                                  duration    = False)   
        audios_Train, audios_Test, _, _, duration_audio, _ = generate_train_and_test_sets(path_base   = path_data, 
                                                                                          patient_val = [patient],
                                                                                          exercise_s  = exercise, 
                                                                                          duration    = False)         
        print("==========================================================================================")
        print("Validating Patient {}: Video Duration Frames:{} Audio Duration Frames:{}".format(patient, duration_video,duration_audio))
   

        videos_train        = {idx: videos[idx] for idx in videos.keys() if idx!=patient}
        labels_train        = {idx: labels[idx] for idx in labels.keys() if idx!=patient}
        samples_type_train  = {idx: samples_type[idx] for idx in samples_type.keys() if idx!=patient}
        exercises_s_train   = {idx: exercises_s[idx] for idx in exercises_s.keys() if idx!=patient}
        repetitions_s_train = {idx: repetitions_s[idx] for idx in repetitions_s.keys() if idx!=patient}

        videos_test        = {patient:videos[patient]}
        labels_test        = {patient:labels[patient]}
        samples_type_test  = {patient:samples_type[patient]}
        exercises_s_test   = {patient:exercises_s[patient]}
        repetitions_s_test = {patient:repetitions_s[patient]}        

        """
        #----------------------------------------------------------------
        # Generate 2D data to train and validate the model
        #----------------------------------------------------------------
        transformations = transforms.Compose([To_Tensor_video_2D()])
        train_data      = VisualDataset2D(names_videos   = videos_Train,
                                          duration       = duration_video,
                                          transform      = transformations)
        test_data       = VisualDataset2D(names_videos   = videos_Test,
                                          duration       = duration_video,
                                          transform      = transformations)

        print('Training samples: {}'.format(train_data.__len__()))
        print('Test samples: {}'.format(test_data.__len__()))

        video_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        video_test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)        
        """

        #----------------------------------------------------------------
        # Generate Video Data Augmentation
        #----------------------------------------------------------------
        video_transformations = transforms.Compose([
        ApplyVideoTransforms(),  # Aplicar transformaciones al video
        To_Tensor_video()  # Pasar a tensor
        ])
        
        #----------------------------------------------------------------
        # Generate 3D video data to train
        #----------------------------------------------------------------
        transformations = transforms.Compose([To_Tensor_video()])
        train_data      = VisualDataset_v2(videos        = videos_train,
                                           labels        = labels_train,
                                           samples_type  = samples_type_train, 
                                           exercises_s   = exercises_s_train, 
                                           repetitions_s = repetitions_s_train,
                                           transform     = video_transformations)
        
        test_data       = VisualDataset_v2(videos        = videos_test,
                                           labels        = labels_test,
                                           samples_type  = samples_type_test, 
                                           exercises_s   = exercises_s_test,
                                           repetitions_s = repetitions_s_test,
                                           transform     = transformations)

        print('Video Training samples: {}'.format(train_data.__len__()))
        print('Video Test samples: {}'.format(test_data.__len__()))

        video_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        video_test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)   

        #----------------------------------------------------------------
        # Generate Audio Data Augmentation
        #----------------------------------------------------------------
        audio_transformations = transforms.Compose([
            AdditiveGaussianNoise(mean=0.0, std=0.005),  # Añadir ruido
            #AdditiveNoiseFloor(noise_level=0.05),  # Añadir ruido blanco adaptado
            #ApplyAudioTransforms(),  # Aplicar enmascarado de frecuencia y tiempo
            To_Tensor_audio()  # Pasar a tensor
        ])        

        #----------------------------------------------------------------
        # Generate audio data to train 
        #----------------------------------------------------------------
        transformations = transforms.Compose([To_Tensor_audio()])
        train_data      = AudioDataset(names_audios  = audios_Train, #VERIFICAR BIEN LOS ESPECTROGRAMAS Y VER SI SON DICIENTES Y VALIDOS. EVITAR QUE NO HAYA ERRORES
                                        duration     = duration_audio,
                                        transform    = audio_transformations)
        
        os.makedirs('spects', exist_ok=True)
        dump(train_data.X, 'spects/train_data_words_X.joblib')