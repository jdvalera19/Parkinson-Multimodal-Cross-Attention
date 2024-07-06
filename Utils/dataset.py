import os
import torch
import random
import torchaudio
import librosa

import numpy as np

from skimage           import io, exposure
from skimage.transform import resize
from torch.utils.data  import Dataset
from scipy             import signal
from tqdm              import tqdm

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

class To_Tensor_video_2D(object):

    def __call__(self, sample):
        image = np.array(sample['video'])

        image = image.transpose((2,0,1))

        sample['video'] = torch.from_numpy(image)

        return sample

#----------------------------------------------------------------------
# Transform the numpy data in torch data
#----------------------------------------------------------------------
# Parameters: None
# Return: the dictionary with samples
#----------------------------------------------------------------------
class To_Tensor_audio(object):

    def __call__(self, sample):
        image = np.array(sample['audio'])
        #print('To tensor:',image.shape)

        #image = image.transpose((2, 0, 1))
        sample['audio'] = torch.from_numpy(image)

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
# Identify the most low duration in the exercise selected
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def identify_max_AVduration(data_audio, data_video):
    max_duration = 0
    max_frames   = 0 

    for audio in data_audio:
        sig, sr = torchaudio.load(audio)
        if sig[0,:].shape[0] > max_duration:
            max_duration = sig[0,:].shape[0]

    for video in data_video:
        number_of_frames= len(os.listdir(video))
        if number_of_frames > max_frames:
            max_frames = number_of_frames

    return max_duration, max_frames

#----------------------------------------------------------------------
# Function to load the video data for all patients 
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def __get_sample_data__(name):

        type_sample  = name.split('-')[0][0]
        patient      = name.split('-')[0]
        repetition   = name.split('-')[1]
        exercise     = name.split('-')[-1][:-4]

        if type_sample == "P":
            label = 1

        else:
            label = 0

        return label, exercise, patient, repetition

def __load_frames__(frames, video_name, duration):
        loaded_frames = []

        while len(frames) < duration:
            rand_index = np.random.randint(len(frames))
            rand_frame = frames[rand_index]
            frames.insert(rand_index, rand_frame)
        
        while len(frames) > duration:
            rand_index = np.random.randint(len(frames))
            frames.pop(rand_index)

        for frame_index, frame_n in enumerate(frames):
            if frame_index%3 == 0:
                frame = io.imread(video_name + '/' + frame_n, as_gray=True)
                frame = resize(frame, (224, 224), anti_aliasing=True)
                frame = np.expand_dims(frame, 2)
                loaded_frames.append(frame)

        return loaded_frames

def load_video_data(path_base = None, exercise_s = None, duration = None):
    videos        = {}
    labels        = {}
    exercises_s   = {}
    repetitions_s = {}
    samples_type  = {}

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
                    path_modality_videos = path_exercise + '/{}'.format('frames')
                    videos_name = os.listdir(path_modality_videos)
                    patient_videos_path = [path_modality_videos + '/' + name_string for name_string in videos_name]

                    samples_type[patient]   = []
                    videos[patient]         = []
                    labels[patient]         = []
                    exercises_s[patient]    = []
                    repetitions_s[patient]   = []

                    for idx, video in enumerate(patient_videos_path):
                        frames     = os.listdir(video)
                        frames.sort()

                        type_sample = video.split('/')[-1][0]

                        label, exercise, patient, repetition = __get_sample_data__(video.split('/')[-1])
                        
                        loaded_frames = __load_frames__(frames, video, duration)
                        
                        videos[patient].append(loaded_frames)
                        labels[patient].append(label)
                        samples_type[patient].append(type_sample)
                        exercises_s[patient].append(exercise)
                        repetitions_s[patient].append(repetition)

                        print(np.shape(loaded_frames), label, type_sample, exercise, repetition)

    return videos, labels, samples_type, exercises_s, repetitions_s




#----------------------------------------------------------------------
# Divide the avanible data into subset to make training and validation 
# takin into account leave one out cross validation
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def generate_train_and_test_sets(path_base = None, patient_val = None, exercise_s = None, duration = None):
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

    if duration:
        duration_audio, duration_video = identify_max_AVduration(audios_V + audios_T, videos_V + videos_T)

    else:
        duration_audio, duration_video = identify_min_AVduration(audios_V + audios_T, videos_V + videos_T)             
    return audios_T, audios_V, videos_T, videos_V, duration_audio, duration_video

#----------------------------------------------------------------------
# Custom Dataset class to generate the tensors to train a model 
#----------------------------------------------------------------------
# Parameters: 
#-----------------------------------------------------------------------
class VisualDataset(Dataset):
    def __init__(self, 
                 names_videos,
                 duration,
                 duration_size,
                 transform):
        
        self.videos                       = names_videos
        self.transform                    = transform
        self.duration                     = duration//2
        self.duration_size                = duration_size
        self.X, self.Y                    = [], []
        self.samples_type, self.exercises = [], []
        self.patients, self.repetition    = [], []
        
        stream = tqdm(total=len(self.videos), desc = 'loading data')

        with stream as pbar:
            for idx, video in enumerate(self.videos):
                frames     = os.listdir(video)
                frames.sort()

                type_sample = video.split('/')[-1][0]
                self.samples_type.append(type_sample)

                label, exercise, patient, repetition = self.__get_sample_data__(video.split('/')[-1])
                if self.duration_size:
                    loaded_frames = self.__load_frames__(frames, video)
                else:
                    loaded_frames = self.__load_frames_v2__(frames, video)
                
                self.X.append(loaded_frames)
                self.Y.append(label)
                self.patients.append(patient)
                self.exercises.append(exercise)
                self.repetition.append(repetition)

                pbar.update(1)
        
    def __get_sample_data__(self, name):

        type_sample  = name.split('-')[0][0]
        patient      = name.split('-')[0]
        repetition   = name.split('-')[1]
        exercise     = name.split('-')[-1][:-4]

        if type_sample == "P":
            label = 1

        else:
            label = 0

        return label, exercise, patient, repetition
    
    def __load_frames__(self, frames, video_name):
        loaded_frames = []

        if len(frames) < self.duration:
            while len(frames) < self.duration:
                rand_index = np.random.randint(len(frames))
                rand_frame = frames[rand_index]
                frames.insert(rand_index, rand_frame)
            
        else:
            while len(frames) > self.duration:
                rand_index = np.random.randint(len(frames))
                frames.pop(rand_index)

        for frame_index, frame_n in enumerate(frames):
            frame = io.imread(video_name + '/' + frame_n, as_gray=True)
            frame = resize(frame, (224, 224), anti_aliasing=True)
            frame = np.expand_dims(frame, 2)
            loaded_frames.append(frame)

        return loaded_frames
    
    def __load_frames_v2__(self, frames, video_name):
        loaded_frames = []
        middle = len(frames)//2

        for frame_index, frame_n in enumerate(frames[middle-self.duration:middle] + frames[middle:middle+self.duration]):
                frame = io.imread(video_name + '/' + frame_n, as_gray=True)
                frame = resize(frame, (224, 224), anti_aliasing=True)
                frame = exposure.equalize_adapthist(frame, clip_limit=0.03)
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
                  "repetition"   : self.repetition[idx],
                  "exercise"     : self.exercises[idx]}
        
        if self.transform:
            sample = self.transform(sample)
 
        return sample

#----------------------------------------------------------------------
# Custom Dataset class to generate the tensors to train a model 
#----------------------------------------------------------------------
# Parameters: 
#-----------------------------------------------------------------------
class VisualDataset_v2(Dataset):
    def __init__(self, 
                 videos,
                 labels,
                 samples_type, 
                 exercises_s, 
                 repetitions_s,
                 transform):
        
        self.transform                    = transform
        self.X, self.Y                    = [], []
        self.samples_type, self.exercises = [], []
        self.patients, self.repetition    = [], []

        for patient in videos.keys():
            self.X            += videos[patient]
            self.Y            += labels[patient]
            self.patients     += [patient for idx in range(len(labels[patient]))]
            self.exercises    += exercises_s[patient]
            self.repetition   += repetitions_s[patient]
            self.samples_type += samples_type[patient]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'video'        : self.X[idx],
                  'label'        : self.Y[idx],
                  "samples_type" : self.samples_type[idx],
                  "patient_id"   : self.patients[idx],
                  "repetition"   : self.repetition[idx],
                  "exercise"     : self.exercises[idx]}
        
        if self.transform:
            sample = self.transform(sample)
 
        return sample

#----------------------------------------------------------------------
# Custom Dataset class to generate the tensors to train a model 
#----------------------------------------------------------------------
# Parameters: 
#-----------------------------------------------------------------------
class VisualDataset2D(Dataset):
    def __init__(self, 
                 names_videos,
                 duration,
                 transform):
        
        self.videos                       = names_videos
        self.transform                    = transform
        self.duration                     = duration//2
        self.X, self.Y                    = [], []
        self.samples_type, self.exercises = [], []
        self.patients, self.repetition    = [], []
        
        stream = tqdm(total=len(self.videos), desc = 'loading data')

        with stream as pbar:
            for idx, video in enumerate(self.videos):
                frames     = os.listdir(video)
                frames.sort()

                type_sample = video.split('/')[-1][0]
                self.samples_type.append(type_sample)

                label, exercise, patient, repetition = self.__get_sample_data__(video.split('/')[-1])
                loaded_frame = self.__load_frame__(frames, video)
                
                self.X.append(loaded_frame)
                self.Y.append(label)
                self.patients.append(patient)
                self.exercises.append(exercise)
                self.repetition.append(repetition)

                pbar.update(1)
        
    def __get_sample_data__(self, name):

        type_sample  = name.split('-')[0][0]
        patient      = name.split('-')[0]
        repetition   = name.split('-')[1]
        exercise     = name.split('-')[-1][:-4]

        if type_sample == "P":
            label = 1

        else:
            label = 0

        return label, exercise, patient, repetition
    
    def __load_frame__(self, frames, video_name):
        
        random_frame = random.choice(frames)
        frame = io.imread(video_name + '/' + random_frame, as_gray=True)
        frame = resize(frame, (224, 224), anti_aliasing=True)
        frame = np.expand_dims(frame, 2)

        return frame

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'video'        : self.X[idx],
                  'label'        : self.Y[idx],
                  "samples_type" : self.samples_type[idx],
                  "patient_id"   : self.patients[idx],
                  "repetition"   : self.repetition[idx],
                  "exercise"     : self.exercises[idx]}
        
        if self.transform:
            sample = self.transform(sample)
 
        return sample

#----------------------------------------------------------------------
# Custom Dataset class to generate the tensors to train a model 
#----------------------------------------------------------------------
# Parameters: 
#-----------------------------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, 
                 names_audios,
                 duration,
                 transform):
        
        self.audios                       = names_audios
        self.transform                    = transform
        self.duration                     = duration//2
        self.X, self.Y                    = [], []
        self.samples_type, self.exercises = [], []
        self.patients, self.repetition    = [], []
        
        for audio in self.audios:
            sig, sr = torchaudio.load(audio)

            sig         = self.remove_noise(sig)
            sig         = self.crop_signal(sig, self.duration)
            process_sig = self.spectro_gram((sig, sr))

            type_sample = audio.split('/')[-1][0]
            self.samples_type.append(type_sample)

            label, exercise, patient, repetition = self.__get_sample_data__(audio.split('/')[-1])
            
            #print(np.shape(process_sig))
            self.X.append(process_sig)
            self.Y.append(label)
            self.patients.append(patient)
            self.exercises.append(exercise)
            self.repetition.append(repetition)

        #print(np.shape(self.X), np.shape(self.Y))
        
    def __get_sample_data__(self, name):

        type_sample  = name.split('-')[0][0]
        patient      = name.split('-')[0]
        repetition   = name.split('-')[1]
        exercise     = name.split('-')[-1][:-4]

        if type_sample == "P":
            label = 1

        else:
            label = 0

        return label, exercise, patient, repetition
    
    def remove_noise(self, y):
        # Cargar archivo de audio

        # Aplicar un filtro de ruido utilizando una ventana de Hamming
        #y = y.astype(np.float32)
        y_filt = signal.filtfilt(librosa.filters.get_window("hamming", 10), [1], y)

        # Eliminar silencios del audio
        y_filt = librosa.effects.trim(y_filt, top_db=30, frame_length=1024, hop_length=256)[0]
        
        return y_filt
    
    def spectro_gram(self, aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db  = 80

        fr, time, spec = signal.spectrogram(sig,fs=48000.,nperseg=480,noverlap=240,nfft=512)
        spec = 10 * np.log10(spec + 1e-7)

        return spec

    def crop_signal(self, signal, duration):

        middle = len(signal[0,:])//2
        
        duration_middle = duration//2

        crop_signal = signal[:,middle-duration_middle:middle+duration_middle]

        return crop_signal

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'audio'        : self.X[idx],
                  'label'        : self.Y[idx],
                  "samples_type" : self.samples_type[idx],
                  "patient_id"   : self.patients[idx],
                  "repetition"   : self.repetition[idx],
                  "exercise"     : self.exercises[idx]}
        
        if self.transform:
            sample = self.transform(sample)
 
        return sample