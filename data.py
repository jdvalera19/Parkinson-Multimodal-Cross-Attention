import os
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


with open('Models/DataAugmentation_2/Vowels/4HeadAudioVideo015AttnMaps/attention_maps_C0_25.pkl', 'rb') as f:
    attention_outputs = pickle.load(f)        




# Convierte los mapas de atención a CPU y luego a arreglos numpy para guardarlos
for i, attention_map in enumerate(attention_outputs):
    attention_maps_cpu = attention_map.cpu().numpy()  # Mueve a CPU y convierte a numpy
    np.save(f'Models/DataAugmentation_2/Vowels/4HeadAudioVideo015AttnMaps/attention_maps_C0_25_{i}.npy', attention_maps_cpu)  # Guarda cada mapa de atención en un archivo separado

print("Mapas de atención guardados como archivos .npy.")

