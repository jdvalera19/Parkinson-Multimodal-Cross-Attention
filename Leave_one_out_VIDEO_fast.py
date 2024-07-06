import os

from Utils.model2d import CNNModel2D
from Utils.model3d import CNNModel3D

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings

warnings.filterwarnings('ignore')

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
    epoch       = 100
    batch_size  = 5
    exercise    = 'Phonemes'
    path_data   = '/data/franklin_pupils/Jose/Dataset/AudioVisualData_v7'
    note        = 'VIDEO:weights'
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

    
    _, _, _, _, _, duration_video = generate_train_and_test_sets(path_base   = path_data, 
                                                                 patient_val = ['P0'],
                                                                 exercise_s  = exercise, 
                                                                 duration    = s_duration)
    
    videos, labels, samples_type, exercises_s, repetitions_s = load_video_data(path_base   = path_data,
                                                                               exercise_s  = exercise, 
                                                                               duration    = duration_video)

    
    for patient in patients:
        
        """
        _, _, videos_Train, videos_Test, _, duration_video = generate_train_and_test_sets(path_base   = path_data, 
                                                                                          patient_val = [patient],
                                                                                          exercise_s  = exercise, 
                                                                                          duration    = False)        
        """
        print("==========================================================================================")
        print("Validating Patient {}: Duration Frames:{}".format(patient, duration_video))

        
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
        # Generate 3D data to train and validate the model
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

        print('Training samples: {}'.format(train_data.__len__()))
        print('Test samples: {}'.format(test_data.__len__()))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
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

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
        """
        #----------------------------------------------------------------
        # Load network2
        #----------------------------------------------------------------
        model = CNNModel3D() #3d Model
        #model = CNNModel2D() #2d Model
        model.to(device)

        #----------------------------------------------------------------
        # Train and save the model
        #----------------------------------------------------------------
        dataloaders = {"train":train_loader, "test":test_loader}
        model, Y_true, Y_pred, PK_props, C_props, sample_ids, exercises, repetitions = train_model_CE(model       = model,
                                                                                        num_epochs  = epoch,
                                                                                        dataloaders = dataloaders,
                                                                                        modality    = 'video',
                                                                                        lr          = lr)

        Y_true_g        += Y_true
        Y_pred_g        += Y_pred
        PK_props_g      += PK_props
        C_props_g       += C_props
        samples_ids_g   += sample_ids
        exercises_g     += exercises
        repetitions_g   += repetitions
        
    dataframe_of_results_name = 'Results_v2/Note:{}-Lr:{}-Epoch:{}-Exercise:{}-duration_size:{}.csv'.format(note, lr, epoch, exercise, s_duration)

    data_frame_of_results = pd.DataFrame({'Y_true'       : Y_true_g,
                                          'Y_pred'       : Y_pred_g,
                                          'PK_props'     : PK_props_g,
                                          'C_props'      : C_props_g,
                                          'Sample_ids'   : samples_ids_g,
                                          'Exercise_g'   : exercises_g,
                                          'Repetition'   : repetitions_g})

    data_frame_of_results.to_csv(dataframe_of_results_name)

    view_results(dataframe_of_results_name)


