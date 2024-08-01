import os
import shutil
import imageio
import torch
import cv2

import matplotlib.pyplot as plt
import numpy             as np
import torch.nn.functional as F
import torch.nn as nn

import torchvision.models as models
from torchvision.models import VGG16_Weights

from tqdm            import tqdm
from Utils.i3dpt     import I3D, Unit3Dpy
from sklearn.metrics import accuracy_score
from torchmetrics    import Accuracy
from torch.nn.functional import interpolate

from multiattn import Embedding_RFBMultiHAttnNetwork_V4, New_RFBMultiHAttnNetwork_V4, RFBMultiHAttnNetwork_V4



def load_I3D(pre_train = False):
    base_model = I3D(num_classes=400, modality='rgb')
    if pre_train:
        base_model.load_state_dict(torch.load('Models/model_rgb.pth'))

    base_model.conv3d_1a_7x7 = Unit3Dpy(out_channels=64,
                                        in_channels=1,
                                        kernel_size=(7, 7, 7),
                                        stride=(2, 2, 2),
                                        padding='SAME')
    base_model.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024,
                                        out_channels=2,
                                        kernel_size=(1, 1, 1),
                                        activation=None,
                                        use_bias=True,
                                        use_bn=False)

    return base_model

def load_resnet50(pre_train = True, input_channels=1):

    base_model  = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pre_train)
    
    base_model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    base_model.fc    = torch.nn.Linear(2048, 2)

    return base_model

def load_vgg16(pre_train = True, input_channels=1):

    base_model  = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    base_model.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    
    num_features = base_model.classifier[-1].in_features
    features = list(base_model.classifier.children())[:-1] # Elimina la última capa
    features.extend([torch.nn.Linear(num_features, 2)])
    base_model.classifier = torch.nn.Sequential(*features)

    return base_model

class ModifiedVGG16(nn.Module):
    def __init__(self, pre_train=True, input_channels=1):
        super(ModifiedVGG16, self).__init__()
        base_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        base_model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Modificar el clasificador
        num_features = base_model.classifier[-1].in_features
        features = list(base_model.classifier.children())[:-1]  # Elimina la última capa
        features.extend([nn.Linear(num_features, 2)])  # Añadir la nueva última capa lineal
        self.features = base_model.features
        self.classifier = nn.Sequential(*features)
    
    def forward(self, x, extract_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Aplanar las características para el clasificador
        if extract_features:
            # Pasar por todas las capas excepto la última y devolver la salida de la capa 5
            for layer in self.classifier[:-1]:
                x = layer(x)
            return x
        else:
            x = self.classifier(x)
            return x

def load_vgg16_features(pre_train = True, input_channels=1):

    base_model  = models.vgg16(weights='DEFAULT')
    
    base_model.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    base_model.features = torch.nn.Sequential(*list(base_model.features.children()))
    #x = torch.randn(1, input_channels, 224, 224)  # Example input tensor
    #x = base_model.features(input)

    return base_model



 

def load_vgg16_for_embedding(pre_train=True, input_channels=1):
    base_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    #base_model = models.vgg16()
    
    base_model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    num_features = base_model.classifier[0].in_features
    features = list(base_model.classifier.children())[:-1]  # Elimina la última capa para clasificar
    new_classifier = nn.Sequential(
        #nn.Linear(24576, 4096),  # Vowels
        #nn.Linear(20480, 4096),  # Phonemes
        #nn.Linear(8192, 4096),  # Words
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(4096, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(128, 2)
    )    

    base_model.classifier = new_classifier
    
    return base_model

class EmbeddingVGG16(nn.Module):
    def __init__(self, pre_train=True, input_channels=1):
        super(EmbeddingVGG16, self).__init__()
        self.base_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.base_model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        num_features = self.base_model.classifier[-1].in_features
        self.features = self.base_model.features

        # Modificar el clasificador para hacerlo más flexible
        self.classifier = nn.Sequential(
            #nn.Linear(24576, 4096), #Vowels
            nn.Linear(20480, 4096),  # Phonemes
            #nn.Linear(8192, 4096),  # Words
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.final_layer = nn.Linear(128, 2)
        
    def forward(self, x, extract_features=False):
        x = self.features(x)  # Aplicar capas convolucionales
        x = torch.flatten(x, 1)  # Aplanar las características para el clasificador
        x = self.classifier(x)  # Pasar por el clasificador modificado
        if extract_features:
            return x  # Devuelve el embebido justo antes de la última capa lineal
        x = self.final_layer(x)  # Aplicar la última capa lineal
        return x

    def get_features(self, x): #Para obtener el resultado de mi input luego de la extracción de características
        features = self.base_model.features(x)
        return features

def load_vgg16_for_embedding_2(pre_train=True, input_channels=1):
    return EmbeddingVGG16(pre_train=pre_train, input_channels=input_channels)

class CustomVGG16(nn.Module):
    def __init__(self, input_channels=1, final_conv_filters=64):
        super(CustomVGG16, self).__init__()
        self.base_model = load_vgg16(input_channels=input_channels)
        self.base_model.classifier = nn.Sequential(
            #nn.Linear(20480, 4096),  # Phonemes
            #nn.Linear(8192, 4096),  # Words
            nn.Linear(24576, 4096),  # Vowels
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Identity()  # Reemplaza la última capa con nn.Identity
        )

    def forward(self, x):
        features = self.base_model.features(x)
        features = features.view(features.size(0), -1)
        embedding = self.base_model.classifier(features)  # Embedding después de la penúltima capa
        return embedding

    def get_features(self, x):
        return self.base_model.features(x)

#----------------------------------------------------------------------
# Training a model using the binary cross entropy loss
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def save_activations(activations, sample, exercise, repetition, type):
    cm = plt.get_cmap('copper')

    for idx, video in enumerate(sample):
        try:
            os.mkdir('Images/' + video)
        except OSError as error:
            None

        activation = 255.*cm(activations[idx, 0, :, :, :].cpu().detach().numpy())
        activation = activation.astype('uint8')
        imageio.mimwrite('{}/{}-{}-{}-{}.gif'.format('Images/' + video , type, video, repetition[idx], exercise[idx]), activation, 'GIF', duration=0.2)
        


#----------------------------------------------------------------------
# Training a model using cross entropy loss function
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def train_model_CE_VIDEO_get_layer1(model, num_epochs=3, dataloaders=None, modality=None, lr = 0.00001): #si sirvio

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accuracy  = Accuracy(task='BINARY').cuda()

    for epoch in range(num_epochs):
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc  = 0.0

            Y      = []
            Y_pred = []
            PK_props = []
            C_props = []
            Samples = []
            exercises = []
            repetitions = []          

            stream = tqdm(total=len(dataloaders[phase]), desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss, running_acc))

            with stream as pbar:

                for index, data in enumerate(dataloaders[phase]):
                    
                    img        = data[modality].type(torch.float).cuda()
                    labels     = data['label'].cuda()
                    sample     = data['patient_id']
                    repetition = data['repetition']
                    exercise   = data['exercise']

                    features     = model(img,return_features=True)
                # Extract features from the last convolutional layer
                    print(features.size())
                """                
                    loss        = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    running_loss += loss.item()
                    logits       = torch.nn.Softmax(dim=1)(outputs)

                    predicted = logits.max(1).indices
                    Y_pred    += list(predicted.cpu().detach().numpy())
                    Y         += list(labels.cpu().detach().numpy())

                    if phase == 'test':
                        PK_props  += list(logits.cpu().detach().numpy()[:,1])
                        C_props   += list(logits.cpu().detach().numpy()[:,0])
                        Samples   += sample
                        exercises += exercise
                        repetitions += repetition

                        #if epoch + 1 == num_epochs:
                        #    activations_first = model.get_embs_first(img)
                        #    activations_last = model.get_embs_last(img)
                        #    save_activations(activations_first, sample, exercise, repetition, 'first')
                        #    save_activations(activations_last, sample, exercise, repetition, 'last')
                   
                    total_samples = index + 1
                    running_acc = accuracy_score(Y, Y_pred)
                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss/total_samples, running_acc)

                    pbar.update(1)
                    """
    
    return model, Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions, features



def train_model_CE_AUDIO_get_layer(model, num_epochs=3, dataloaders=None, modality=None, lr = 0.00001):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc  = 0.0

            Y      = []
            Y_pred = []
            PK_props = []
            C_props = []
            Samples = []
            exercises = []
            repetitions = []

            stream = tqdm(total=len(dataloaders[phase]), desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss, running_acc))

            with stream as pbar:

                for index, data in enumerate(dataloaders[phase]):
                    
                    img        = data[modality].type(torch.float).cuda()
                    labels     = data['label'].cuda()
                    sample     = data['patient_id']
                    repetition = data['repetition']
                    exercise   = data['exercise']
                    #print(img.size())
                    result = img
                    for i in range(15):
                        result = model.features[i](result)
                    
                    last_conv_output_2 = result
                    #print(last_conv_output_2.size())
                    # Calculate start and end indices for center crop
                    start_idx = (last_conv_output_2.size(2) - 54) // 2
                    end_idx = start_idx + 54

                    # Perform the crop
                    last_conv_output_3 = last_conv_output_2[:, :, start_idx:end_idx, :]

                    print(last_conv_output_3.size())
                    
                    if last_conv_output_3.size(0) != 6:
                        print("aca")
                    """
                    loss        = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    running_loss += loss.item()
                    logits       = torch.nn.Softmax(dim=1)(outputs)

                    predicted = logits.max(1).indices
                    Y_pred    += list(predicted.cpu().detach().numpy())
                    Y         += list(labels.cpu().detach().numpy())

                    if phase == 'test':
                        PK_props  += list(logits.cpu().detach().numpy()[:,1])
                        C_props   += list(logits.cpu().detach().numpy()[:,0])
                        Samples   += sample
                        exercises += exercise
                        repetitions += repetition

                    total_samples = index + 1
                    running_acc = accuracy_score(Y, Y_pred)
                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss/total_samples, running_acc)

                    pbar.update(1)

                    """
    return model, Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions, last_conv_output_3

def train_model_CE_AUDIO_VIDEO_get_layer(audio_model, video_model, num_epochs, audio_dataloaders, video_dataloaders, audio_modality, video_modality, lr, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_audio = torch.optim.Adam(audio_model.parameters(), lr=0.000001)
    optimizer_video = torch.optim.Adam(video_model.parameters(), lr=0.0001)
    #torch.cuda.empty_cache() PROBAR SIN ESTO

    for epoch in range(num_epochs):
        for phase in audio_dataloaders.keys():
            if phase == 'train':
                audio_model.train()
                video_model.train()
            else:
                audio_model.eval()
                video_model.eval()
    
            running_loss = 0.0
            running_acc  = 0.0

            Y      = []
            Y_pred = []
            PK_props = []
            C_props = []
            Samples = []
            exercises = []
            repetitions = []
            v = 1
    
            stream = tqdm(total=len(audio_dataloaders[phase]), desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss, running_acc))

            with stream as pbar:

                for index, (audio_data, video_data) in enumerate(zip(audio_dataloaders[phase], video_dataloaders[phase])):
                    
                    img_audio        = audio_data[audio_modality].type(torch.float).cuda()
                    img_video        = video_data[video_modality].type(torch.float).cuda()                    
                    labels     = audio_data['label'].cuda()
                    sample     = audio_data['patient_id']
                    repetition = audio_data['repetition']
                    exercise   = audio_data['exercise']

                    result = img_audio
                    embedding_audio = audio_model(result)
                    #for i in range(15): #para extraer características
                    #    result = audio_model.features[i](result)
                    
                    #last_conv_output_2 = result #features de audio
                    #features_video     = video_model(img_video,return_features=False) #Vector de características
                    embedding_video = video_model.get_embedding(img_video) #Embebido de video
                    #print(last_conv_output_2.size())
                    # Calculate start and end indices for center crop
                    #start_idx = (last_conv_output_2.size(2) - 54) // 2
                    #end_idx = start_idx + 54

                    # Perform the crop
                    #features_audio = last_conv_output_2[:, :, start_idx:end_idx, :]
                    #features_audio = last_conv_output_2[:, :features_video.size(1), :54, :]
                    
                    #Perform a resize
                    #conv1x1 = torch.nn.Conv2d(last_conv_output_2.size(1), embedding_video.size(1), kernel_size=1).cuda()
                    #features_audio = conv1x1(last_conv_output_2)    
                    #features_audio = F.interpolate(features_audio, size=(embedding_video.size(2), embedding_video.size(3)), mode='bilinear', align_corners=False)
                    #current_width = features_video.size(3)
                    #start_idx = (features_video.size(3) - features_audio.size(3)) // 2
                    #end_idx = start_idx + features_audio.size(3)
                    #features_video = features_video[:,:,:,start_idx:end_idx]

                    #print(features_audio.size())
                    #print(features_video.size())
                    context_dim = embedding_video.size(1)
                    #query_dim = features_audio.size(1)
                    query_dim = embedding_audio.size(1)
                    #context_dim = features_video.size(1)
                    filters_head = 1 #1 cabeza acá?
                    cross_model = Embedding_RFBMultiHAttnNetwork_V4(query_dim=query_dim, context_dim=context_dim,
                                filters_head=filters_head)
                    
                    cross_model.to(device)
                    outputs = cross_model(embedding_audio, embedding_video)
                    #print(outputs.shape)
                    loss        = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer_audio.zero_grad()
                        optimizer_video.zero_grad()
                        loss.backward()
                        optimizer_audio.step()
                        optimizer_video.step()
                        
                    running_loss += loss.item()
                    logits       = torch.nn.Softmax(dim=1)(outputs)

                    predicted = logits.max(1).indices
                    Y_pred    += list(predicted.cpu().detach().numpy())
                    Y         += list(labels.cpu().detach().numpy())

                    if phase == 'test':
                        PK_props  += list(logits.cpu().detach().numpy()[:,1])
                        C_props   += list(logits.cpu().detach().numpy()[:,0])
                        Samples   += sample
                        exercises += exercise
                        repetitions += repetition

                    total_samples = index + 1
                    running_acc = accuracy_score(Y, Y_pred)
                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss/total_samples, running_acc)

                    pbar.update(1)

    # Guardar los pesos del cross_model al finalizar el entrenamiento
    torch.save(cross_model.state_dict(), 'Models/atenciónEmbebidos_AUDIO_VIDEO3D_256LINEAR:weights-Lr:1e-05-Epoch:50-Exercise:Vowels-duration_size:False.pth')                    


    #Cargar los pesos
    #cross_model = RFBMultiHAttnNetwork_V4(query_dim=query_dim, context_dim=context_dim, filters_head=filters_head)
    #cross_model.load_state_dict(torch.load('cross_model_weights.pth'))
    #cross_model.to(device)
    
    return Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions

def adapt_state_dict(loaded_state_dict):
    new_state_dict = {}
    for key, value in loaded_state_dict.items():
        # Añade 'base_model.' antes de cada clave si no está ya presente
        new_key = 'base_model.' + key if not key.startswith('base_model.') else key
        new_state_dict[new_key] = value
    return new_state_dict

def train_model_CE_AUDIO_VIDEO_WEIGHTS(audio_model, video_model, num_epochs, audio_dataloaders, video_dataloaders, audio_modality, video_modality, lr, device, patient):
    criterion = torch.nn.CrossEntropyLoss()
    # Cargar los pesos para cada paciente específico
    audio_weight_path = f'./Models/AudioPhonemes/{patient}.pth'
    video_weight_path = f'./Models/VideoPhonemes/{patient}.pth'
    video_model.load_state_dict(torch.load(video_weight_path))
    audio_model.load_state_dict(torch.load(audio_weight_path))
    audio_model.eval()
    video_model.eval()

    # Inicializar el modelo de atención
    cross_model = Embedding_RFBMultiHAttnNetwork_V4(query_dim=128, context_dim=128, filters_head=1)
    cross_model.to(device)
    optimizer_cross = torch.optim.Adam(cross_model.parameters(), lr=lr)

    #if audio_model.training and video_model.training == True:
    #    audio_model.eval()
    #    video_model.eval()

    for epoch in range(num_epochs):
        for phase in audio_dataloaders.keys():
            if phase == 'train':
                cross_model.train()
            else:
                cross_model.eval()

            running_loss = 0.0
            running_acc  = 0.0
            Y      = []
            Y_pred = []
            PK_props = []
            C_props = []
            Samples = []
            exercises = []
            repetitions = []
            v = 1
    
            stream = tqdm(total=len(audio_dataloaders[phase]), desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss, running_acc))

            with torch.set_grad_enabled(phase == 'train'):
                for index, (audio_data, video_data) in enumerate(zip(audio_dataloaders[phase], video_dataloaders[phase])):
                        
                    img_audio        = audio_data[audio_modality].type(torch.float).cuda()
                    img_video        = video_data[video_modality].type(torch.float).cuda()                    
                    labels     = audio_data['label'].cuda()

                        
                    embedding_audio = audio_model(img_audio, extract_features=True)
                    embedding_video = video_model.get_embedding(img_video) #Embebido de video
                    outputs = cross_model(embedding_audio, embedding_video)
                    loss        = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer_cross.zero_grad()
                        loss.backward()
                        optimizer_cross.step()
                            
                    running_loss += loss.item()
                    logits       = torch.nn.Softmax(dim=1)(outputs)

                    predicted = logits.max(1).indices
                    Y.extend(labels.cpu().numpy())
                    Y_pred.extend(predicted.cpu().numpy())

                    if phase == 'test':
                        PK_props.extend(logits[:, 1].cpu().numpy())
                        C_props.extend(logits[:, 0].cpu().numpy())
                        Samples.extend(audio_data['patient_id'])
                        exercises.extend(audio_data['exercise'])
                        repetitions.extend(audio_data['repetition'])

                    running_acc = accuracy_score(Y, Y_pred)
                    stream.set_description('Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss/(index+1), running_acc))
                    stream.update()

    # Guardar los pesos del cross_model al finalizar el entrenamiento
    torch.save(cross_model.state_dict(), f'./Models/AudioVideoPhonemes/{patient}.pth')
    
    return Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions


def load_weights_for_patient(model, base_path, patient_id):
    model_path = f"{base_path}/{patient_id}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Asegúrate de que el modelo esté en modo de evaluación
        print(f"Weights loaded from {model_path}")
    else:
        print(f"No weights found at {model_path}")
    return model



def train_model_CE_AUDIO_VIDEO_repeat(audio_model, video_model, num_epochs=3, audio_dataloaders=None, video_dataloaders=None, audio_modality=None, video_modality=None, lr=0.00001, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_audio = torch.optim.Adam(audio_model.parameters(), lr=lr)
    optimizer_video = torch.optim.Adam(video_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for phase in audio_dataloaders.keys():
            if phase == 'train':
                audio_model.train()
                video_model.train()
            else:
                audio_model.eval()
                video_model.eval()

            running_loss = 0.0
            running_acc = 0.0

            Y = []
            Y_pred = []
            PK_props = []
            C_props = []
            Samples = []
            exercises = []
            repetitions = []

            stream = tqdm(total=len(audio_dataloaders[phase]), desc='Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch + 1, num_epochs, phase, running_loss, running_acc))

            with stream as pbar:
                for index, (audio_data, video_data) in enumerate(zip(audio_dataloaders[phase], video_dataloaders[phase])):
                    img_audio = audio_data[audio_modality].type(torch.float).to(device)
                    img_video = video_data[video_modality].type(torch.float).to(device)
                    labels = audio_data['label'].to(device)
                    sample = audio_data['patient_id']
                    repetition = audio_data['repetition']
                    exercise = audio_data['exercise']

                    # Extracción de características de audio
                    features_audio = audio_model.get_features(img_audio)

                    # Extracción de características de video
                    features_video = video_model(img_video, return_features=True)  # Vector de características de video

                    # Manejo de la variación en el tamaño del lote
                    batch_size = features_audio.size(0)  # Obtiene el tamaño del lote actual

                    # Verificar dimensiones antes de continuar
                    #print(f"features_audio shape: {features_audio.shape}")
                    #print(f"features_video shape: {features_video.shape}")

                    # Expandir features_audio para que tenga dimensiones compatibles con features_video
                    # Necesitamos repetir features_audio en las dimensiones espaciales (últimas dos dimensiones)
                    # para hacerlo del tamaño [5, 512, 1, 7, 7]

                    # Reducir la dimensión temporal usando la media para obtener una representación promedio a lo largo del tiempo
                    compressed_audio = features_audio.mean(dim=2, keepdim=True)  # [5, 512, 1, 2]

                    # Primero, expandimos la última dimensión para llegar a 7
                    expanded_audio = compressed_audio.repeat(1, 1, 1, 7)  # [5, 512, 1, 14]

                    # Para lograr la dimensión [5, 512, 1, 7, 7], necesitamos manejar tanto las dimensiones de anchura como de altura
                    # Redimensionar correctamente de [5, 512, 1, 14] a [5, 512, 1, 7, 2], luego promediar sobre la última dimensión
                    expanded_audio = expanded_audio.reshape(batch_size, 512, 1, 7, 2).mean(dim=-1)  # [5, 512, 1, 7]
                    
                    # Asegurando que el redimensionamiento se ajusta al tamaño del lote
                    #expanded_audio = interpolate(expanded_audio.unsqueeze(-1), size=(7, 7), mode='nearest')
                    
                    # Ahora necesitamos repetir la dimensión de altura para obtener [5, 512, 1, 7, 7]
                    expanded_audio = expanded_audio.unsqueeze(-1).repeat(1, 1, 1, 1, 7)  # [5, 512, 1, 7, 7]
                    
                    # Definir una capa convolucional para ajustar los canales de expanded_audio a 64
                    conv1x1 = nn.Conv3d(512, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1)).cuda()
                    
                    expanded_audio = conv1x1(expanded_audio)
                    # Ahora expanded_audio tiene la forma deseada [5, 64, 1, 7, 7] y es compatible con features_video
                    #print("Expanded Audio Shape:", expanded_audio.shape)
                    
                    # Multiplicar elemento a elemento
                    vector_combinado = expanded_audio * features_video

                    # Re-multiplicar por el vector de video original
                    vector_final = vector_combinado * features_video
                    #print("Vector final Shape:", vector_final.shape)

                    model = AudioVideoClassifier(input_dim=3136, hidden_dim=128, output_dim=2)
                    model.to(device)
                    outputs = model(vector_final)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer_audio.zero_grad()
                        optimizer_video.zero_grad()
                        loss.backward()
                        optimizer_audio.step()
                        optimizer_video.step()

                    running_loss += loss.item()
                    logits = torch.nn.Softmax(dim=1)(outputs)

                    predicted = logits.max(1).indices
                    Y_pred += list(predicted.cpu().detach().numpy())
                    Y += list(labels.cpu().detach().numpy())

                    if phase == 'test':
                        PK_props += list(logits.cpu().detach().numpy()[:, 1])
                        C_props += list(logits.cpu().detach().numpy()[:, 0])
                        Samples += sample
                        exercises += exercise
                        repetitions += repetition

                    total_samples = index + 1
                    running_acc = accuracy_score(Y, Y_pred)
                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch + 1, num_epochs, phase, running_loss / total_samples, running_acc)

                    pbar.update(1)

    # Guardar los pesos del modelo combinado al finalizar el entrenamiento
    #torch.save(conv1x1.state_dict(), 'Models/Note:Nueva_atención_MultiplicaciónVectoresDeCaracterísticas_VIDEO3D:weights-Lr:1e-05-Epoch:50-Exercise:Words-duration_size:False.pth')

    return Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions

def train_model_CE_AUDIO_VIDEO_repeat2(audio_model, video_model, num_epochs=3, audio_dataloaders=None, video_dataloaders=None, audio_modality=None, video_modality=None, lr=0.00001, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_audio = torch.optim.Adam(audio_model.parameters(), lr=lr)
    optimizer_video = torch.optim.Adam(video_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for phase in audio_dataloaders.keys():
            if phase == 'train':
                audio_model.train()
                video_model.train()
            else:
                audio_model.eval()
                video_model.eval()

            running_loss = 0.0
            running_acc = 0.0

            Y = []
            Y_pred = []
            PK_props = []
            C_props = []
            Samples = []
            exercises = []
            repetitions = []

            stream = tqdm(total=len(audio_dataloaders[phase]), desc='Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch + 1, num_epochs, phase, running_loss, running_acc))

            with stream as pbar:
                for index, (audio_data, video_data) in enumerate(zip(audio_dataloaders[phase], video_dataloaders[phase])):
                    img_audio = audio_data[audio_modality].type(torch.float).to(device)
                    img_video = video_data[video_modality].type(torch.float).to(device)
                    labels = audio_data['label'].to(device)
                    sample = audio_data['patient_id']
                    repetition = audio_data['repetition']
                    exercise = audio_data['exercise']

                    # Extracción de características de audio
                    features_audio = audio_model.get_features(img_audio)

                    # Extracción de características de video
                    features_video = video_model(img_video, return_features=True)  # Vector de características de video

                    # Manejo de la variación en el tamaño del lote
                    batch_size = features_audio.size(0)  # Obtiene el tamaño del lote actual

                    conv1 = nn.Conv2d(512, 96, kernel_size=1).cuda()  # Reducir canales
                    #conv2 = nn.Conv2d(96, 96, kernel_size=(4, 2), stride=(4, 2)).cuda()  # Reducir tamaño espacial Phonemes 
                    #conv2 = nn.Conv2d(96, 96, kernel_size=(4, 1), stride=(4, 1)).cuda()  # Reducir tamaño espacial, preservar anchura Words
                    conv2 = nn.Conv2d(96, 96, kernel_size=(4, 3), stride=(4, 2)).cuda()  # Vowels
                    # Capa de pooling para ajustar la dimensión de anchura
                    features_audio = conv1(features_audio)
                    features_audio = conv2(features_audio)
                    
                    # Multiplicar elemento a elemento
                    vector_combinado = features_audio * features_video

                    # Re-multiplicar por el vector de video original
                    vector_final = vector_combinado * features_video
                    #print("Vector final Shape:", vector_final.shape)
                    input_dim = 96 * 2 * 2  # Adjust this based on the actual output size of conv2
                    model = AudioVideoClassifier(input_dim=input_dim, hidden_dim=128, output_dim=2)
                    model.to(device)
                    outputs = model(vector_final)

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer_audio.zero_grad()
                        optimizer_video.zero_grad()
                        loss.backward()
                        optimizer_audio.step()
                        optimizer_video.step()

                    running_loss += loss.item()
                    logits = torch.nn.Softmax(dim=1)(outputs)

                    predicted = logits.max(1).indices
                    Y_pred += list(predicted.cpu().detach().numpy())
                    Y += list(labels.cpu().detach().numpy())

                    if phase == 'test':
                        PK_props += list(logits.cpu().detach().numpy()[:, 1])
                        C_props += list(logits.cpu().detach().numpy()[:, 0])
                        Samples += sample
                        exercises += exercise
                        repetitions += repetition

                    total_samples = index + 1
                    running_acc = accuracy_score(Y, Y_pred)
                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch + 1, num_epochs, phase, running_loss / total_samples, running_acc)

                    pbar.update(1)

    # Guardar los pesos del modelo combinado al finalizar el entrenamiento
    #torch.save(conv1x1.state_dict(), 'Models/Note:Nueva_atención_MultiplicaciónVectoresDeCaracterísticas_VIDEO3D:weights-Lr:1e-05-Epoch:50-Exercise:Words-duration_size:False.pth')

    return Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions



class AudioVideoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AudioVideoClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, vector_final):
        # Asumiendo que combined_features ya es el resultado de la multiplicación de audio y video
        vector_final_flat = vector_final.view(vector_final.size(0), -1)
        x = self.relu(self.fc1(vector_final_flat))
        x = self.softmax(self.fc2(x))
        return x





def train_model_CE(model, num_epochs, dataloaders, modality, lr, patient_id, exercise):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accuracy  = Accuracy(task='BINARY').cuda()

    for epoch in range(num_epochs):
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc  = 0.0

            Y      = []
            Y_pred = []
            PK_props = []
            C_props = []
            Samples = []
            exercises = []
            repetitions = []

            stream = tqdm(total=len(dataloaders[phase]), desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss, running_acc))

            with stream as pbar:

                for index, data in enumerate(dataloaders[phase]):
                    
                    img        = data[modality].type(torch.float).cuda()
                    labels     = data['label'].cuda()
                    sample     = data['patient_id']
                    repetition = data['repetition']
                    exercise   = data['exercise']

                    outputs     = model(img)
                    loss        = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    running_loss += loss.item()
                    logits       = torch.nn.Softmax(dim=1)(outputs)

                    predicted = logits.max(1).indices
                    Y_pred    += list(predicted.cpu().detach().numpy())
                    Y         += list(labels.cpu().detach().numpy())

                    if phase == 'test':
                        PK_props  += list(logits.cpu().detach().numpy()[:,1])
                        C_props   += list(logits.cpu().detach().numpy()[:,0])
                        Samples   += sample
                        exercises += exercise
                        repetitions += repetition

                        #if epoch + 1 == num_epochs:
                            #activations_first = model.get_embs_first(img)
                            #activations_last = model.get_embs_last(img)
                            #save_activations(activations_first, sample, exercise, repetition, 'first')
                            #save_activations(activations_last, sample, exercise, repetition, 'last')

                    total_samples = index + 1
                    running_acc = accuracy_score(Y, Y_pred)
                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss/total_samples, running_acc)

                    pbar.update(1)
    # Guarda los pesos después de entrenar todas las épocas para un paciente
    torch.save(model.state_dict(), f'./Models/VideoWords/{patient_id}.pth')
    return model, Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions

def train_model_CE_AUDIO(model, num_epochs, dataloaders, modality, lr, patient_id, exercise):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc  = 0.0

            Y      = []
            Y_pred = []
            PK_props = []
            C_props = []
            Samples = []
            exercises = []
            repetitions = []

            stream = tqdm(total=len(dataloaders[phase]), desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss, running_acc))

            with stream as pbar:

                for index, data in enumerate(dataloaders[phase]):
                    
                    img        = data[modality].type(torch.float).cuda()
                    labels     = data['label'].cuda()
                    sample     = data['patient_id']
                    repetition = data['repetition']
                    exercise   = data['exercise']

                    outputs     = model(img)
                    loss        = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    running_loss += loss.item()
                    logits       = torch.nn.Softmax(dim=1)(outputs)

                    predicted = logits.max(1).indices
                    Y_pred    += list(predicted.cpu().detach().numpy())
                    Y         += list(labels.cpu().detach().numpy())

                    if phase == 'test':
                        PK_props  += list(logits.cpu().detach().numpy()[:,1])
                        C_props   += list(logits.cpu().detach().numpy()[:,0])
                        Samples   += sample
                        exercises += exercise
                        repetitions += repetition

                    total_samples = index + 1
                    running_acc = accuracy_score(Y, Y_pred)
                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss/total_samples, running_acc)

                    pbar.update(1)
    # Guarda los pesos después de entrenar todas las épocas para un paciente
    torch.save(model.state_dict(), f'./Models/AudioWords/{patient_id}.pth')    

    return model, Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions