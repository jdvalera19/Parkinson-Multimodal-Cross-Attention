import os
import random
import numpy as np
import torch
from torchsummary import summary
import matplotlib.pyplot as plt

import pandas as pd

from torchvision import transforms
import torchaudio.transforms as audio_transforms
from torch.utils.data import DataLoader

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
    lr          = 0.00001
    epoch       = 70
    batch_size  = 1
    exercise    = 'Vowels'
    path_data   = '/data/arumota_pupils/Jose/AudioVisualData_v7'
    #path_data   = '/data/franklin_pupils/Jose/Dataset/AudioVisualData_v7'
    note        = 'PRUEBA_ATENCION_FINAL_NoDataAugmentation_1Cabeza0.5AtenciónFinal_atenciónEmbebidos_AUDIO2D_VIDEO3D_PesosGuardados:weights' 
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

    
    concat_model = EmbeddingConcatenation(input_dim=256).to(device)
    optimizer_concat = torch.optim.Adam(concat_model.parameters(), lr=lr)
    
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
        #ApplyVideoTransforms(),  # Aplicar transformaciones al video
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
            #AdditiveGaussianNoise(mean=0.0, std=0.005),  # Añadir ruido
            #AdditiveNoiseFloor(noise_level=0.05),  # Añadir ruido blanco adaptado
            #ApplyAudioTransforms(),  # Aplicar enmascarado de frecuencia y tiempo
            To_Tensor_audio()  # Pasar a tensor
        ])        

        #----------------------------------------------------------------
        # Generate audio data to train 
        #----------------------------------------------------------------
        transformations = transforms.Compose([To_Tensor_audio()])
        train_data      = AudioDataset(names_audios  = audios_Train,
                                        duration     = duration_audio,
                                        transform    = audio_transformations)
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
        #video_model = CNNModel2D()
        video_model = CNNModel3D() #Video
        video_model.to(device)
        #print(video_model.conv_layer2)

        audio_model = CNNModel2D() #Audio
        audio_model.to(device)     
          
                
        video_dataloaders = {"train":video_train_loader, "test":video_test_loader}
        audio_dataloaders = {"train":audio_train_loader, "test":audio_test_loader}

        """
        _, _, _, _, _, _, _, _, features_audio = train_model_CE_AUDIO_get_layer(model       = audio_model,
                                                       num_epochs  = epoch,
                                                       dataloaders = audio_dataloaders,
                                                       modality    = 'audio',
                                                       lr          = lr) 

    
        _, _, _, _, _, _, _, _,features_video = train_model_CE_VIDEO_get_layer1(model       = video_model,
                                                       num_epochs  = epoch,
                                                       dataloaders = video_dataloaders,
                                                       modality    = 'video',
                                                       lr          = lr)  
        """
        Y_true, Y_pred, PK_props, C_props, sample_ids, exercises, repetitions = train_model_CE_AUDIO_VIDEO_WEIGHTS_CONCATENACION(
                                                        audio_model       = audio_model,
                                                        video_model       = video_model,
                                                        concat_model      = concat_model,
                                                        optimizer_concat  = optimizer_concat,  
                                                        num_epochs  = epoch,
                                                        audio_dataloaders = audio_dataloaders,
                                                        video_dataloaders = video_dataloaders,
                                                        audio_modality   =  'audio',
                                                        video_modality    = 'video',
                                                        lr          = lr,
                                                        device = device,
                                                        patient = patient) 

        # Plotting the validation loss against learning rate
        #plt.figure(figsize=(10, 5))
        #plt.plot(lr_history, val_loss, marker='o')
        #plt.xscale('log')
        #plt.xlabel('Learning Rate')
        #plt.ylabel('Validation Loss')
        #plt.title('Validation Loss as a Function of Learning Rate')
        #plt.grid(True)
        #plt.show()
        #plt.savefig(f'Images/validation_loss_plot_{patient}.png')  # Save to file  
        #plt.close()  # Close the plot frame to free resources
        
        Y_true_g        += Y_true
        Y_pred_g        += Y_pred
        PK_props_g      += PK_props
        C_props_g       += C_props
        samples_ids_g   += sample_ids
        exercises_g     += exercises
        repetitions_g   += repetitions


    dataframe_of_results_name = 'Results_v2/NoDataAugmentation/Vowels/Note:{}-Lr:{}-Epoch:{}-Exercise:{}-duration_size:{}.csv'.format(note, lr, epoch, exercise, s_duration)

    data_frame_of_results = pd.DataFrame({'Y_true'       : Y_true_g,
                                          'Y_pred'       : Y_pred_g,
                                          'PK_props'     : PK_props_g,
                                          'C_props'      : C_props_g,
                                          'Sample_ids'   : samples_ids_g,
                                          'Exercise_g'   : exercises_g,
                                          'Repetition'   : repetitions_g})

    data_frame_of_results.to_csv(dataframe_of_results_name)

    view_results(dataframe_of_results_name)
        

        