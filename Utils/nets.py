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

from tqdm            import tqdm
from Utils.i3dpt     import I3D, Unit3Dpy
from sklearn.metrics import accuracy_score
from torchmetrics    import Accuracy

from multiattn import Embedding_RFBMultiHAttnNetwork_V4, RFBMultiHAttnNetwork_V4


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

def load_vgg16_features(pre_train = True, input_channels=1):

    base_model  = models.vgg16(weights='DEFAULT')
    
    base_model.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    base_model.features = torch.nn.Sequential(*list(base_model.features.children()))
    #x = torch.randn(1, input_channels, 224, 224)  # Example input tensor
    #x = base_model.features(input)

    return base_model



def load_vgg16(pre_train = True, input_channels=1):

    base_model  = models.vgg16(weights='DEFAULT')
    
    base_model.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    
    num_features = base_model.classifier[-1].in_features
    features = list(base_model.classifier.children())[:-1] # Elimina la última capa
    features.extend([torch.nn.Linear(num_features, 2)])
    base_model.classifier = torch.nn.Sequential(*features)

    return base_model

class CustomVGG16(nn.Module): #para obtener embebido
    def __init__(self, input_channels=2, final_conv_filters=64):
        super(CustomVGG16, self).__init__()
        self.base_model = load_vgg16(input_channels=input_channels)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(8192, 4096),  # Ajusta este valor según la salida de 'features.view'
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Identity()  # Reemplaza la última capa con nn.Identity
        )

    def forward(self, x):
        features = self.base_model.features(x)
        features = features.view(features.size(0), -1)
        print(features.size())
        embedding = self.base_model.classifier(features)  # Embedding después de la penúltima capa
        return embedding
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

def train_model_CE_AUDIO_VIDEO_get_layer(audio_model, video_model, num_epochs=3, audio_dataloaders=None, video_dataloaders=None, audio_modality=None, video_modality=None, lr = 0.00001, device=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_audio = torch.optim.Adam(audio_model.parameters(), lr=lr)
    optimizer_video = torch.optim.Adam(video_model.parameters(), lr=lr)
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
    torch.save(cross_model.state_dict(), 'Models/Note:2_atenciónEmbebidos_VIDEO3D-1x1_RESIZE_AUDIO:weights-Lr:1e-05-Epoch:50-Exercise:Words-duration_size:False.pth')                    


    #Cargar los pesos
    #cross_model = RFBMultiHAttnNetwork_V4(query_dim=query_dim, context_dim=context_dim, filters_head=filters_head)
    #cross_model.load_state_dict(torch.load('cross_model_weights.pth'))
    #cross_model.to(device)
    
    return Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions




#----------------------------------------------------------------------
# Training a model using cross entropy loss function
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def train_model_CE_AUDIO(model, num_epochs=3, dataloaders=None, modality=None, lr = 0.00001):

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
            # List to store features
            features = []            

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
    
    return model, Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions, features



def train_model_CE(model, num_epochs=3, dataloaders=None, modality=None, lr = 0.00001):

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

                        if epoch + 1 == num_epochs:
                            activations_first = model.get_embs_first(img)
                            activations_last = model.get_embs_last(img)
                            save_activations(activations_first, sample, exercise, repetition, 'first')
                            save_activations(activations_last, sample, exercise, repetition, 'last')

                    total_samples = index + 1
                    running_acc = accuracy_score(Y, Y_pred)
                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss/total_samples, running_acc)

                    pbar.update(1)
    
    return model, Y, Y_pred, PK_props, C_props, Samples, exercises, repetitions

