import os
import cv2
import torch
import shutil
import subprocess

import numpy as np

from facenet_pytorch import MTCNN
from PIL             import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

path_data = '/data/Datasets/Parkinson/Facial_Expression_Dataset/AudioVisualDataset_v4/Videos_face'
classes   = os.listdir(path_data)

print('Founded classes are:', classes)

folder_to_save_data = '/home/brayan/AudioVisualData_v1'
get_face            = False
detector            = MTCNN(select_largest = False,
                            min_face_size  = 10,
                            thresholds     = [0.6, 0.7, 0.7],
                            post_process   = False,
                            image_size     = 224,
                            device         = device)

try:
    os.mkdir(folder_to_save_data)

except OSError as error:
    shutil.rmtree(folder_to_save_data)
    os.mkdir(folder_to_save_data)

for class_ in classes:
    path_class     = path_data + '/{}'.format(class_)
    new_path_class = folder_to_save_data + '/{}'.format(class_)
    os.mkdir(new_path_class)
    patients       = os.listdir(path_class)
    
    for patient in patients:
        path_patient     = path_class + '/{}'.format(patient)
        new_path_patient = new_path_class + '/{}'.format(patient)
        os.mkdir(new_path_patient)
        exercises        = os.listdir(path_patient)
        
        for exercise in exercises:
            path_exercise     = path_patient + '/{}'.format(exercise)
            new_path_exercise = new_path_patient + '/{}'.format(exercise)
            os.mkdir(new_path_exercise)
            videos            = os.listdir(path_exercise)
            
            path_audio  = new_path_exercise + '/{}'.format('audio') 
            path_frames = new_path_exercise + '/{}'.format('frames') 
            os.mkdir(path_audio)
            os.mkdir(path_frames)
            
            for video in videos:
                
                os.mkdir(path_frames + '/{}'.format(video))
                path_video = path_exercise + '/{}'.format(video)
                audio_out = path_audio + '/' + video[:-4] + '.mp3'
                frames_out = path_frames + '/{}'.format(video) + '/%06d.jpg'
                subprocess.call(["ffmpeg", "-i", path_video, "-vn", "-acodec", "mp3", audio_out])
                subprocess.call(["ffmpeg", "-i", path_video, "-vf", "fps=60", frames_out])

                if get_face:
                    frames_path = '/'.join(frames_out.split('/')[:-1])
                    frames_names = os.listdir(frames_path)
                    
                    for frame_name in frames_names:
                        img = Image.open('{}/{}'.format(frames_path, frame_name))
                        detection = detector.forward(img)

                        if detection != None:
                            detection = detection.permute(1, 2, 0).int().numpy()
                            r = np.expand_dims(detection[:,:,2], axis=2)
                            g = np.expand_dims(detection[:,:,1], axis=2)
                            b = np.expand_dims(detection[:,:,0], axis=2)
                            detection = np.concatenate((r,g,b,), axis=2)
                            cv2.imwrite('{}/{}'.format(frames_path, frame_name), detection)
                        
                        else:

                            os.remove('{}/{}'.format(frames_path, frame_name))
                    
print('Data saved at folder {}'.format(folder_to_save_data))
