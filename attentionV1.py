import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

import pandas as pd

from torchvision      import transforms
from torch.utils.data import DataLoader

#Custom librarian
from Utils.dataset       import *
from Utils.nets          import *
from Utils.visualization import *


if __name__== "__main__":

    #-------------------------------------------------------------------
    # Define the model parameters
    #-------------------------------------------------------------------
    lr          = 0.00001
    epoch       = 2
    batch_size  = 5
    exercise    = 'Words'
    path_data   = '/home/Data/franklin_pupils/Jose/Dataset/AudioVisualData_v7'
    note        = 'VIDEO:weights' #change the word VIDEO
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

    audios_Train, audios_Test, videos_T, videos_V, duration_audio, duration_video = generate_train_and_test_sets(path_base   = path_data, 
                                                                 patient_val = ['P0'],
                                                                 exercise_s  = exercise, 
                                                                 duration    = s_duration)
    
    videos, labels, samples_type, exercises_s, repetitions_s = load_video_data(path_base   = path_data,
                                                                               exercise_s  = exercise, 
                                                                               duration    = duration_video)
    

    for patient in patients[0:3]:
        print("==========================================================================================")
        print("Validating Patient {}: Duration Frames:{}".format(patient, duration_audio)) 
    
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

        #----------------------------------------------------------------
        # Generate video data to train
        #----------------------------------------------------------------
        transformations = transforms.Compose([To_Tensor_video()])
        train_data      = VisualDataset_v2(videos        = videos_train,
                                           labels        = labels_train,
                                           samples_type  = samples_type_train, 
                                           exercises_s   = exercises_s_train, 
                                           repetitions_s = repetitions_s_train,
                                           transform     = transformations)
        
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
        # Generate audio data to train 
        #----------------------------------------------------------------
        transformations = transforms.Compose([To_Tensor_audio()])
        train_data      = AudioDataset(names_audios  = audios_Train,
                                        duration     = duration_audio,
                                        transform    = transformations)
        test_data       = AudioDataset(names_audios  = audios_Test,
                                        duration     = duration_audio,
                                        transform    = transformations)

        print('Audio Training samples: {}'.format(train_data.__len__()))
        print('Audio Test samples: {}'.format(test_data.__len__()))

        audio_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        audio_test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)           

        #----------------------------------------------------------------
        # Load both networks
        #----------------------------------------------------------------
        #video_model = load_I3D(pre_train=True)
        #video_model.to(device)

        audio_model = load_vgg16(pre_train = True, input_channels=2)
        audio_model.to(device)     
        
        video_dataloaders = {"train":video_train_loader, "test":video_test_loader}
        features, _, _, _, _, _, _, _ = train_model_CE_VIDEO_get_layer(model       = video_model,
                                                       num_epochs  = epoch,
                                                       dataloaders = video_dataloaders,
                                                       modality    = 'video',
                                                       lr          = lr)  

        print(features)                                                             


