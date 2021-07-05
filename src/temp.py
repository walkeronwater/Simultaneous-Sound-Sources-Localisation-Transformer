import numpy as np
sigPair = np.load('/content/drive/MyDrive/SSSL/npy/array0.npy')
print(sigPair.shape)
sigPairList = []
sigPairList.append(sigPair)
# del sigPair


"""### Extract binaural cues

- IPD cues (similar as ITD (ambiguous at high frequencies))

  arg(S_l/S_r)
  
  
- Spectral cues (essentially tells about ILD cues)

  do STFT and then convert it to magnitude and phase
"""

def cartesian2euler(val):
    x = val.real
    y = val.imag
    
    r = np.sqrt(x**2+y**2)

    theta = np.arctan(
        np.divide(y, x, where=x!=0)
    )
    # if x != 0:
    #     theta = np.arctan(y/x)
    # else:
    #     theta = np.pi/2
        
    return normalise(r), normalise(theta)

def calIPD(seqL, seqR):
    temp = np.divide(seqL, seqR, out=np.zeros_like(seqL), where=seqR!=0)
    ipd = np.arctan(np.divide(np.imag(temp), np.real(temp), out=np.zeros_like(np.real(temp)), where=np.real(temp)!=0))
    return ipd

def normalise(seq):
    return seq/np.linalg.norm(seq)

temp = np.array([[1, 2, 3],[1,2,3]])
print(normalise(temp))

from scipy import signal
import random

def binauralCues(sigPair, fs):
    f, t, Zxx = signal.stft(sigPair[0, 0, 0], fs, nperseg=1023)
    # spectralCues = np.zeros(sigPair.shape[:-2] + (Zxx.shape[1], Zxx.shape[0]) + (4,), dtype='float')
    # ipdCues = np.zeros(sigPair.shape[:-2] + (Zxx.shape[1], Zxx.shape[0]), dtype='float')
    cues = np.zeros(sigPair.shape[:-2] + (Zxx.shape[1], Zxx.shape[0]) + (5,), dtype='float')

    del f, t, Zxx

    for i in range(sigPair.shape[0]):
        for locIndex in range(sigPair.shape[1]):
            f_l, t_l, Zxx_l = signal.stft(sigPair[i, locIndex, 0], fs, nperseg=1023)
            f_r, t_r, Zxx_r = signal.stft(sigPair[i, locIndex, 1], fs, nperseg=1023)

            r_l, theta_l = cartesian2euler(Zxx_l)
            r_r, theta_r = cartesian2euler(Zxx_r)

            # ipdCues[i, locIndex] = normalise(np.transpose(calIPD(Zxx_l, Zxx_r), (1, 0)))
            # spectralCues[i, locIndex] = np.transpose(np.array([r_l, theta_l, r_r, theta_r]), (2, 1 ,0))
            cues[i, locIndex] = np.concatenate(
                (np.expand_dims(
                    normalise(np.transpose(calIPD(Zxx_l, Zxx_r), (1, 0))), axis=-1
                    ),
                 np.transpose(np.array([r_l, theta_l, r_r, theta_r]), (2, 1 ,0))
                 ),
                 axis=-1
            )
    return cues
    # return ipdCues, spectralCues

print(sigPair.shape)
cues = binauralCues(sigPair, 16000)

print(cues.shape)

"""### Tensorisation"""

import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import class_weight
from torch.autograd import Variable
import torch.nn.functional as F

# ipd = torch.from_numpy(ipdCues.astype(np.float32))
# slMag = torch.from_numpy(spectralCues[:,:,:,:,0].astype(np.float32))
# slPhase = torch.from_numpy(spectralCues[:,:,:,:,1].astype(np.float32))
# srMag = torch.from_numpy(spectralCues[:,:,:,:,2].astype(np.float32))
# srPhase = torch.from_numpy(spectralCues[:,:,:,:,3].astype(np.float32))

cues_ = torch.from_numpy(cues.astype(np.float32))


Nloc = cues_.shape[1]
labels = np.zeros((cues_.shape[0], cues_.shape[1], 1)).astype(np.long)
for i in range(cues_.shape[0]):
    for j in range(cues_.shape[1]):
        labels[i,j] = i

labels_ = torch.from_numpy(labels)
labels_ = labels_.reshape(labels_.shape[0]*labels_.shape[1], 1)

cues_ = cues_.reshape(cues_.shape[0] * cues_.shape[1], cues_.shape[2], cues_.shape[3], cues_.shape[4])


print("cues shape: ",cues.shape)
print("cues shape: ",cues_.shape)
print("labels shape",labels.shape)
print("labels_ shape: ",labels_.shape)
# ipd_ = ipd.reshape(ipd.shape[0]*ipd.shape[1], ipd.shape[2], ipd.shape[3])
# slMag_ = slMag.reshape(slMag.shape[0]*slMag.shape[1], slMag.shape[2], slMag.shape[3])
# slPhase_ = slPhase.reshape(slPhase.shape[0]*slPhase.shape[1], slPhase.shape[2], slPhase.shape[3])
# srMag_ = srMag.reshape(srMag.shape[0]*srMag.shape[1], srMag.shape[2], srMag.shape[3])
# srPhase_ = srPhase.reshape(srPhase.shape[0]*srPhase.shape[1], srPhase.shape[2], srPhase.shape[3])
# print(cues_.shape)
# print(cues_[5,0,0:10])

# dataSet = torch.stack([ipd_, slMag_, slPhase_, srMag_, srPhase_]).permute(1, 2, 3, 0)

train_data = TensorDataset(cues_, labels_)

batch_size = 32
Ntrain = round(0.6*cues_.shape[0])
if Ntrain % batch_size == 1:
    Ntrain -=1
Nvalid = round(0.2*cues_.shape[0])
if Nvalid % batch_size == 1:
    Nvalid -=1
Ntest = cues_.shape[0] - Ntrain - Nvalid
if Ntest % batch_size == 1:
    Ntest -=1
print("Dataset separation: ",Ntrain, Nvalid, Ntest)

train, valid, test = torch.utils.data.random_split(train_data, [Ntrain, Nvalid, Ntest], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset=valid, batch_size=32, shuffle=True, num_workers=0)

"""## Transformer

### Model 1

From scratch
"""

class SelfAttention(nn.Module):
    def __init__(self, freqSize, heads):
        super(SelfAttention, self).__init__()
        self.freqSize = freqSize
        self.heads = heads
        self.head_dim = freqSize // heads

        # assert debug
        assert (
            self.head_dim * heads == freqSize
        ), "Embedding size needs to be divisible by heads"

        # obtain Q K V matrices by linear transformation
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, freqSize)

    def forward(self, value, key, query):
        # Get number of training examples
        N = query.shape[0]

        value_time, value_freq = value.shape[1], value.shape[2]
        key_time, key_freq = key.shape[1], key.shape[2]
        query_time, query_freq = query.shape[1], query.shape[2]

        # Split the embedding into self.heads different pieces
        value = value.reshape(N, value_time, self.heads, self.head_dim)
        key = key.reshape(N, key_time, self.heads, self.head_dim)
        query = query.reshape(N, query_time, self.heads, self.head_dim)

        values = self.values(value)  # (N, value_len, heads, head_dim)
        keys = self.keys(key)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.freqSize ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_time, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class Attention(nn.Module):
    def __init__(self, embedSize, locSize):
        super(Attention, self).__init__()
        self.embedSize = embedSize

        # obtain Q K V matrices by linear transformation
        self.values = nn.Linear(embedSize, embedSize, bias=False)
        self.keys = nn.Linear(embedSize, embedSize, bias=False)
        self.queries = nn.Linear(embedSize, embedSize, bias=False)
        self.fc_out = nn.Linear(embedSize, locSize)

    def forward(self, value, key, query):
        # Get number of training examples
        N = query.shape[0]

        value_time, value_freq = value.shape[1], value.shape[2]
        key_time, key_freq = key.shape[1], key.shape[2]
        query_time, query_freq = query.shape[1], query.shape[2]

        values = self.values(value)  # (N, value_len, heads, head_dim)
        keys = self.keys(key)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        energy = torch.einsum("ntqe,ntke->neqk", [queries, keys])

        attention = torch.softmax(energy / (self.embedSize ** (1 / 2)), dim=3)

        out = torch.einsum("neqk,ntve->ntqe", [attention, values]).reshape(
            N, query_time, query_freq, self.embedSize
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, freqSize, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(freqSize, heads)
        self.norm1 = nn.LayerNorm(freqSize)
        self.norm2 = nn.LayerNorm(freqSize)

        self.feed_forward = nn.Sequential(
            nn.Linear(freqSize, forward_expansion * freqSize),
            nn.ReLU(),
            nn.Linear(forward_expansion * freqSize, freqSize),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        freqSize, # frequency bins
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.freqSize = freqSize
        self.device = device

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    freqSize,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        timeSize, freqSize = x.shape[-2], x.shape[-1]
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(x)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out)

        return out

class SSSL(nn.Module):
    def __init__(
    self,
    locSize,
    timeSize, # time windows
    freqSize, # frequency bins
    num_layers,
    heads,
    device,
    forward_expansion,
    dropout,
    isDebug
    ):
        super(SSSL, self).__init__()
        self.encoder = Encoder(           
            freqSize, # frequency bins
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
        )
        self.attention = Attention(5, locSize)
        self.fcFreq = nn.Linear(freqSize, 1)
        self.fcTime = nn.Linear(timeSize, 1)
        self.isDebug = isDebug
        # self.softmaxLayer = nn.Softmax(dim = -1)
    def forward(self, cues_):
        enc_ipd = self.encoder(cues_[:,:,:,0])
        if self.isDebug == True:
            print("enc_ipd shape: ",enc_ipd.shape)
        enc_slMag = self.encoder(cues_[:,:,:,1])
        enc_slPhase = self.encoder(cues_[:,:,:,2])
        enc_srMag = self.encoder(cues_[:,:,:,3])
        enc_srPhase = self.encoder(cues_[:,:,:,4])
        enc = torch.stack([enc_ipd, enc_slMag, enc_slPhase, enc_srMag, enc_srPhase])
        # del enc_ipd, enc_slMag, enc_slPhase, enc_srMag, enc_srPhase
        enc = enc.permute(1,2,3,0)
        # encCat = torch.cat((enc_ipd, enc_slMag, enc_slPhase, enc_srMag, enc_srPhase), -1)
        # print(enc.shape)
        if self.isDebug == True:
            print("enc shape: ",enc.shape)

        attOut = self.attention(enc, enc, enc)
        if self.isDebug == True:
            print("attOut shape: ",attOut.shape)

        out = self.fcFreq(attOut.permute(0,1,3,2))
        out = out.squeeze(-1)

        # out = torch.mean(attOut, -2)
        # out = out.squeeze(-1)
        if self.isDebug == True:
            print("FC freq shape: ",out.shape)
        
        out = self.fcTime(out.permute(0, 2, 1))
        out = out.squeeze(-1)

        # out = torch.mean(out, -2)
        # out = out.squeeze(-2)
        if self.isDebug == True:
            print("FC time shape: ",out.shape)


        # out = self.softmaxLayer(out)
        return out

Nfreq = cues_.shape[2]
print(Nfreq)
Ntime = cues_.shape[1]
print(Ntime)
# Nloc = cues_.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numLayers = 6
model = SSSL(Nloc, Ntime, Nfreq, numLayers, 8, device, 4, 0, True).to(device)
testInput = cues_[-18:-1].to(device)
testLabel = labels_[-18:-1].to(device)
testOutput = model(testInput)
print(testInput.shape)
print(testOutput.shape)

criterion = nn.CrossEntropyLoss()
loss = criterion(testOutput, testLabel.squeeze(1))
print(loss)
loss.backward()

print(cues_[-18:-1])
print(labels_[-18:-1])




"""### Network training"""

# Commented out IPython magic to ensure Python compatibility.
# import gc

# gc.collect()
# torch.cuda.empty_cache()

'''num_epochs = 100
learning_rate = 1e-4
# batch_size = 32
early_epoch = 100
new_early_epoch = 0
new_val_loss = 0.0

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, verbose=True)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
numLayers = 6
model = SSSL(Nloc, Ntime, Nfreq, numLayers, 8, device, 4, 0, False).to(device)

for epoch in range(num_epochs):
    print("\nEpoch %d" % (epoch + 1))
    model.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0

    for i, data in enumerate(train_loader, 0):
        length = len(train_loader)
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels.long()).to(device)

        
        # ipd = inputs[:,:,:,0]
        # slMag = inputs[:,:,:,1]
        # slPhase = inputs[:,:,:,2]
        # srMag = inputs[:,:,:,3]
        # srPhase = inputs[:,:,:,4]
        
        # outputs = model(ipd)
        outputs = model(inputs)

        print("Input shape: ",inputs.shape)
        print("Ouput shape: ", outputs.shape)
        print("Label shape: ", labels.shape)
        loss = criterion(outputs, labels.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.squeeze(1).data).sum().item()
    print("correct:",correct)
    print("total",total)
    print('Training Loss: %.04f | Training Acc: %.4f%% '
#         % (sum_loss / (i + 1), 100.0 * correct / total))
    

    val_loss = 0.0
    val_correct = 0.0
    val_total = 0.0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels.long()).to(device)

            # ipd = inputs[:,:,:,0]
            # slMag = inputs[:,:,:,1]
            # slPhase = inputs[:,:,:,2]
            # srMag = inputs[:,:,:,3]
            # srPhase = inputs[:,:,:,4]
            
            outputs = model(inputs)
            val_loss = criterion(outputs, labels.squeeze(1))
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels.squeeze(1).data).sum().item()
        scheduler.step(val_loss)

    print('Val_Loss: %.04f | Val_Acc: %.4f%% '
#         % (val_loss, 100.0 * val_correct / val_total))

    if (100.0 * val_correct / val_total <= new_val_loss):
        new_early_epoch += 1
    else:
        new_val_loss = 100.0 * val_correct / val_total
        new_early_epoch = 0
    if (new_early_epoch >= early_epoch):
        break

# loss = criterion(outputs, labels)
for i, data in enumerate(train_loader, 0):
    print(i,' ',len(data),'',data[0].shape,' ',data[1].shape)
'''
