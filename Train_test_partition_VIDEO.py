import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
    lr                = 0.000001
    epoch             = 25
    batch_size        = 2
    exercise          = 'Words'
    path_data_test    = '/home/brayan/AudioVisualData_v1'
    path_data_train   = '/home/brayan/AudioVisualData_v2'
    note              = 'TTP_data_v2_vs_v1'

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

    _, _, videos_Train, _, _, min_duration_video_train = generate_train_and_test_sets(path_base=path_data_train, patient_val=[], exercise_s=exercise)
    _, _, videos_Test, _, _, min_duration_video_test   = generate_train_and_test_sets(path_base=path_data_test, patient_val=[], exercise_s=exercise)

    #----------------------------------------------------------------
    # Generate data to train and validate the model
    #----------------------------------------------------------------
    transformations = transforms.Compose([To_Tensor_video()])
    train_data      = VisualDataset(names_videos = videos_Train,
                                    duration     = min_duration_video_train,
                                    transform    = transformations)
    test_data       = VisualDataset(names_videos = videos_Test,
                                    duration     = min_duration_video_test,
                                    transform    = transformations)

    print('Training samples: {}'.format(train_data.__len__()))
    print('Test samples: {}'.format(test_data.__len__()))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    #----------------------------------------------------------------
    # Load network2
    #----------------------------------------------------------------
    model = load_I3D(pre_train=False)
    model.to(device)

    #----------------------------------------------------------------
    # Train and save the model
    #----------------------------------------------------------------
    dataloaders = {"train":train_loader, "test":test_loader}
    model, Y_true, Y_pred, PK_props, C_props, sample_ids, exercises = train_model_CE(model       = model,
                                                                                     num_epochs  = epoch,
                                                                                     dataloaders = dataloaders,
                                                                                     modality    = 'video',
                                                                                     lr          = lr)


    dataframe_of_results_name = 'Results/Note:{}-Lr:{}-Epoch:{}-Exercise:{}.csv'.format(note, lr, epoch, exercise)

    data_frame_of_results = pd.DataFrame({'Y_true'       : Y_true,
                                          'Y_pred'       : Y_pred,
                                          'PK_props'     : PK_props,
                                          'C_props'      : C_props,
                                          'Sample_ids'   : sample_ids,
                                          'Exercise_g'   : exercises})

    data_frame_of_results.to_csv(dataframe_of_results_name)

    view_results(dataframe_of_results_name)


