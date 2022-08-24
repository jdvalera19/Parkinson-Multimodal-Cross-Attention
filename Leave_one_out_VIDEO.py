import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import warnings

warnings.filterwarnings('ignore')

import torch

from torchvision      import transforms
from torch.utils.data import DataLoader

#Custom librarian
from Utils.dataset import *
from Utils.nets    import *

if __name__== "__main__":

    #-------------------------------------------------------------------
    # Define the model parameters
    #-------------------------------------------------------------------
    lr                = 0.00001
    epoch             = 25
    batch_size        = 2
    exercise          = 'Words'
    path_data         = '/home/brayan/AudioVisualData'
    note              = 'LOO_data_v2'

    #-------------------------------------------------------------------
    # Select the GPU to improve the evaluation stage
    #-------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('-------------------------------------------------------------------')
        print("Using GPU for training:", torch.cuda.get_device_name(), '-')
        print('-------------------------------------------------------------------')
    else:
        device = torch.device("cpu")
        print('-------------------------------------------------------------------')
        print("Failed to find GPU, using CPU instead.-")
        print('-------------------------------------------------------------------')

    #-------------------------------------------------------------------
    # List with all patients and list to save the prediction per patient
    #-------------------------------------------------------------------
    parkinson_patients = ["P{}".format(idx) for idx in [0,1,2,3,4]]
    control_patients   = ["C{}".format(idx) for idx in [0,1,2]]
    patients           = control_patients + parkinson_patients

    Y_true_g       = []
    Y_pred_g       = []
    PK_props_g     = []
    C_props_g      = []
    samples_ids_g  = []
    exercises_g    = []

    for patient in patients:
        _, _, videos_T, videos_V, _, min_duration_video = generate_train_and_test_sets(path_base=path_data, patient_val=[patient], exercise_s=exercise)
        print("==========================================================================================")
        print("Validating Patient {}: Duration Frames:{}".format(patient, min_duration_video))

        #----------------------------------------------------------------
        # Generate data to train and validate the model
        #----------------------------------------------------------------
        transformations = transforms.Compose([To_Tensor_video()])
        train_data      = VisualDataset(names_videos = videos_T,
                                        duration     = min_duration_video,
                                        transform    = transformations)
        test_data       = VisualDataset(names_videos = videos_V,
                                        duration     = min_duration_video,
                                        transform    = transformations)

        print('Training samples: {}'.format(train_data.__len__()))
        print('Test samples: {}'.format(test_data.__len__()))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

        #----------------------------------------------------------------
        # Load network
        #----------------------------------------------------------------
        model = load_I3D()
        model.to(device)

        #----------------------------------------------------------------
        # Train and save the model
        #----------------------------------------------------------------
        dataloaders = {"train":train_loader, "test":test_loader}
        model, Y, Y_pred, PK_props, C_props, Sample_ids, exercises = train_model_CE(model       = model,
                                                                                    num_epochs  = epoch,
                                                                                    dataloaders = dataloaders,
                                                                                    modality    = 'video',
                                                                                    lr          = lr)

