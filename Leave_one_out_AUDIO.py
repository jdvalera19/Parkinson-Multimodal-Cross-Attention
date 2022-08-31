import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch

import pandas as pd

from torchvision      import transforms
from torch.utils.data import DataLoader

#Custom librarian
from Utils.dataset       import *
from Utils.nets          import *
from Utils.visualization import *

if __name__== "__main__":
    #----------------------------------------------------------------
    # Define the model parameters
    #----------------------------------------------------------------
    lr                = 0.0000002
    epoch             = 75
    batch_size        = 2
    exercise          = 'Words'
    path_data         = '/home/brayan/AudioVisualData_v2'
    note              = 'AUDIO:LOO_data_v2'

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

    for patient in patients:
        audios_Train, audios_Test, _, _, min_duration_audio, _ = generate_train_and_test_sets(path_base=path_data, patient_val=[patient], exercise_s=exercise)
        print("==========================================================================================")
        print("Validating Patient {}: Duration Frames:{}".format(patient, min_duration_audio))

        #----------------------------------------------------------------
        # Generate data to train and validate the model
        #----------------------------------------------------------------
        transformations = transforms.Compose([To_Tensor_audio()])
        train_data      = AudioDataset(names_audios  = audios_Train,
                                        duration     = min_duration_audio,
                                        transform    = transformations)
        test_data       = AudioDataset(names_audios  = audios_Test,
                                        duration     = min_duration_audio,
                                        transform    = transformations)

        print('Training samples: {}'.format(train_data.__len__()))
        print('Test samples: {}'.format(test_data.__len__()))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

        #----------------------------------------------------------------
        # Load network2
        #----------------------------------------------------------------
        model =load_resnet50(pre_train = True)
        model.to(device)

        #----------------------------------------------------------------
        # Train and save the model
        #----------------------------------------------------------------
        dataloaders = {"train":train_loader, "test":test_loader}
        model, Y_true, Y_pred, PK_props, C_props, sample_ids, exercises = train_model_CE(model  = model,
                                                                                    num_epochs  = epoch,
                                                                                    dataloaders = dataloaders,
                                                                                    modality    = 'audio',
                                                                                    lr          = lr)

        Y_true_g        += Y_true
        Y_pred_g        += Y_pred
        PK_props_g      += PK_props
        C_props_g       += C_props
        samples_ids_g   += sample_ids
        exercises_g     += exercises
    
    dataframe_of_results_name = 'Results/Note:{}-Lr:{}-Epoch:{}-Exercise:{}.csv'.format(note, lr, epoch, exercise)

    data_frame_of_results = pd.DataFrame({'Y_true'       : Y_true_g,
                                          'Y_pred'       : Y_pred_g,
                                          'PK_props'     : PK_props_g,
                                          'C_props'      : C_props_g,
                                          'Sample_ids'   : samples_ids_g,
                                          'Exercise_g'   : exercises_g})

    data_frame_of_results.to_csv(dataframe_of_results_name)

    view_results(dataframe_of_results_name)
