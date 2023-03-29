import os
import cv2
import torch
import shutil
import subprocess

import numpy as np

from mtcnn           import MTCNN
from PIL             import Image

path_data = '/data/Datasets/Parkinson/Facial_Expression_Dataset/AudioVisualDataset_v7'
classes   = os.listdir(path_data)

print('Founded classes are:', classes)

folder_to_save_data = '/home/brayan/AudioVisualData_v7'
get_face            = True
detector            = MTCNN()

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
                    frames_names = sorted(os.listdir(frames_path))

                    for image_p in frames_names:
                        img = cv2.imread('{}/{}'.format(frames_path, image_p))
                        detection = detector.detect_faces(img)
                        if len(detection)>0:
                            box = detection[0]['box']
                            break
                    
                    for frame_name in frames_names:
                        img = cv2.imread('{}/{}'.format(frames_path, frame_name))
                        img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
                        
                        try:
                            cv2.imwrite('{}/{}'.format(frames_path, frame_name), img)
                        except:
                            os.remove('{}/{}'.format(frames_path, frame_name))
print('Data saved at folder {}'.format(folder_to_save_data))
