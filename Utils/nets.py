import torch

from tqdm            import tqdm
from Utils.i3dpt     import I3D, Unit3Dpy
from sklearn.metrics import accuracy_score


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


#----------------------------------------------------------------------
# Training a model using cross entropy loss function
#----------------------------------------------------------------------
# Parameters: 
# Return: 
#-----------------------------------------------------------------------
def train_model_CE(model, num_epochs=3, dataloaders=None, modality=None, lr = 0.00001):

    criterion   = torch.nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    
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

            stream = tqdm(total=len(dataloaders[phase]), desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss, running_acc))

            with stream as pbar:

                for index, data in enumerate(dataloaders[phase]):
                    
                    img    = data[modality].type(torch.float).cuda()
                    labels = data['label'].cuda()
                    sample = data['patient_id']
                    exercise = data['exercise']

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


                    total_samples = index + 1

                    running_acc = accuracy_score(Y, Y_pred)

                    stream.desc = 'Epoch {}/{}-{}-loss:{:.4f}-acc:{:.4f}'.format(epoch+1, num_epochs, phase, running_loss/total_samples, running_acc)

                    pbar.update(1)
    
    return model, Y, Y_pred, PK_props, C_props, Samples, exercises



