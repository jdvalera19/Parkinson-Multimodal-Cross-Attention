from cProfile import label
import os
import torch
import torchaudio

import numpy as np

from skimage          import io
from torch.utils.data import Dataset

#----------------------------------------------------------------------
# Transform the numpy data in torch data
#----------------------------------------------------------------------
# Parameters: None
# Return: the dictionary with samples
#----------------------------------------------------------------------
class To_Tensor_video(object):

    def __call__(self, sample):
        image = np.array(sample['video'])

        image = image.transpose((3,0,1,2))

        sample['video'] = torch.from_numpy(image)

        return sample

#----------------------------------------------------------------------
# Identify the most low duration in the exercise selected
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def identify_min_AVduration(data_audio, data_video):
    min_duration = np.Inf
    min_frames   = np.Inf 

    for audio in data_audio:
        sig, sr = torchaudio.load(audio)
        if sig[0,:].shape[0] < min_duration:
            min_duration = sig[0,:].shape[0]

    for video in data_video:
        number_of_frames= len(os.listdir(video))
        if number_of_frames < min_frames:
            min_frames = number_of_frames

    return min_duration, min_frames

#----------------------------------------------------------------------
# Divide the avanible data into subset to make training and validation 
# takin into account leave one out cross validation
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def generate_train_and_test_sets(path_base = None, patient_val = None, exercise_s = None):

    videos_T = []
    audios_T = []
    
    videos_V = []
    audios_V = []

    classes = os.listdir(path_base)
    
    for class_ in classes:
        path_class = path_base + '/{}'.format(class_)
        patients = os.listdir(path_class)

        for patient_idx, patient in enumerate(patients):
            path_patient     = path_class + '/{}'.format(patient)
            exercises        = os.listdir(path_patient)

            for exercise in exercises:
                if exercise_s == exercise:
                    path_exercise         = path_patient + '/{}'.format(exercise)

                    path_modality_frames = path_exercise + '/{}'.format('frames')
                    path_modality_audio = path_exercise + '/{}'.format('audio')
                    data = os.listdir(path_modality_frames)

                    if patient in patient_val:
                        videos_V += [path_modality_frames + '/' + name_string for name_string in data]
                        audios_V += [path_modality_audio + '/' + name_string[:-4] + '.mp3' for name_string in data]
                    else:
                        videos_T += [path_modality_frames + '/' + name_string for name_string in data]
                        audios_T += [path_modality_audio + '/' + name_string[:-4] + '.mp3' for name_string in data]

    min_duration_audio, min_duration_video = identify_min_AVduration(audios_V + audios_T, videos_V + videos_T)
                        
    return audios_T, audios_V, videos_T, videos_V, min_duration_audio, min_duration_video

#----------------------------------------------------------------------
# Custom Dataset class to generate the tensors to train a model 
#----------------------------------------------------------------------
# Parameters: 
#-----------------------------------------------------------------------
class VisualDataset(Dataset):
    def __init__(self, 
                 names_videos,
                 duration,
                 transform):
        
        self.videos                       = names_videos
        self.transform                    = transform
        self.duration                     = duration//2
        self.X, self.Y                    = [], []
        self.samples_type, self.exercises = [], []
        self.patients                     = []
        
        for video in self.videos:
            frames     = os.listdir(video)
            frames.sort()

            type_sample = video.split('/')[-1][0]
            self.samples_type.append(type_sample)

            label, exercise, patient = self.__get_sample_data__(video.split('/')[-1])
            loaded_frames = self.__load_frames__(frames, video)
            
            self.X.append(loaded_frames)
            self.Y.append(label)
            self.patients.append(patient)
            self.exercises.append(exercise)
        
    def __get_sample_data__(self, name):

        type_sample = name.split('-')[0][0]
        patient      = name.split('-')[0]
        exercise    = name.split('-')[-1][:-4]

        if type_sample == "P":
            label = 1

        else:
            label = 0

        return label, exercise, patient
    
    def __load_frames__(self, frames, video_name):
        loaded_frames = []
        middle = len(frames)//2

        for frame_index, frame_n in enumerate(frames[middle-self.duration:middle] + frames[middle:middle+self.duration]):
                frame = io.imread(video_name + '/' + frame_n, as_gray=True)
                frame = np.expand_dims(frame, 2)
                loaded_frames.append(frame)

        return loaded_frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'video'        : self.X[idx],
                  'label'        : self.Y[idx],
                  "samples_type" : self.samples_type[idx],
                  "patient_id"   : self.patients[idx],
                  "exercise"     : self.exercises[idx]}
        
        if self.transform:
            sample = self.transform(sample)
 
        return sample