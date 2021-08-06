import torch
import torch.nn as nn
from graphviz import Source
from torchviz import make_dot

from utils_model import *

class SelfAttention(nn.Module):
    """
    Self attention module for (time, freq) dependency learning
    """
    def __init__(self, Nfreq, heads):
        super(SelfAttention, self).__init__()
        self.Nfreq = Nfreq
        self.heads = heads
        self.head_dim = Nfreq // heads

        # assert debug
        assert (
            self.head_dim * heads == Nfreq
        ), "Embedding size needs to be divisible by heads"

        # obtain Q K V matrices by linear transformation
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, Nfreq)

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

        attention = torch.softmax(energy / (self.Nfreq ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_time, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class Attention(nn.Module):
    """
    Self attention module for (freq, cues) dependency learning 
    """
    def __init__(self, embedSize):
        super(Attention, self).__init__()
        self.embedSize = embedSize

        # obtain Q K V matrices by linear transformation
        self.values = nn.Linear(embedSize, embedSize, bias=False)
        self.keys = nn.Linear(embedSize, embedSize, bias=False)
        self.queries = nn.Linear(embedSize, embedSize, bias=False)
        self.fc_out = nn.Linear(embedSize, embedSize)

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
    def __init__(self, Nfreq, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(Nfreq, heads)
        self.norm1 = nn.LayerNorm(Nfreq)
        self.norm2 = nn.LayerNorm(Nfreq)

        self.feed_forward = nn.Sequential(
            nn.Linear(Nfreq, forward_expansion * Nfreq),
            nn.ReLU(),
            nn.Linear(forward_expansion * Nfreq, Nfreq),
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
        Ntime,
        Nfreq,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout
    ):

        super(Encoder, self).__init__()
        self.Nfreq = Nfreq
        self.device = device

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    Nfreq,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        # fixed positional encoding
        # self.positional_encodings = torch.linspace(0, 1, Ntime)
        # self.positional_encodings = self.positional_encodings.repeat((batchSize, Nfreq, 1))
        # self.positional_encodings = self.positional_encodings.permute(0, 2, 1).to(device)
        # learnable postional embedding
        # self.position_embedding = nn.Embedding(max_length, Nfreq)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, Ntime, Nfreq = x.shape

        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        positional_encodings = torch.linspace(0, 1, Ntime)
        positional_encodings = positional_encodings.repeat((N, Nfreq, 1))
        positional_encodings = positional_encodings.permute(0, 2, 1).to(self.device)
        # print(f"positional encoding shape: {positional_encodings.shape}")
        # print(positional_encodings[0,:,0])
        
        out = x + positional_encodings
        out = self.dropout(out)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out)

        return out

class EncoderDIYTransformer(nn.Module):
    def __init__(
        self,
        task,
        Ntime,
        Nfreq,
        Ncues,
        Nsound,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        isDebug
    ):
        super(EncoderDIYTransformer, self).__init__()
        self.encoder = Encoder(
            Ntime,
            Nfreq,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout
        )
    def forward(self, inputs):
        encList = []
        for i in range(inputs.shape[-1]):
            enc = self.encoder(inputs[:, :, :, i].permute(0, 2, 1))
            encList.append(enc)
        out = torch.stack(encList)
        out = out.permute(1, 2, 3, 0)
        return out

class FCModules(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        activation,
        dropout
    ):
        super(FCModules, self).__init__()
        if activation.lower() == "tanh":
            act_func = nn.Tanh()
        elif activation.lower() == "relu":
            act_func = nn.ReLU()

        self.FC_blocks = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            act_func,
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            act_func,
            nn.Linear(256, 256),
            act_func,
            nn.Linear(256, output_size, bias=False)
        )

    def forward(self, inputs):
        return self.FC_blocks(inputs)

class DecoderSrcCls(nn.Module):
    """
    Source-specific classifier
    """
    def __init__(
        self,
        enc_out_size,
        Nsound,
        dropout
    ):
        super(DecoderSrcCls, self).__init__()

        self.FC_layers = FCModules(
            input_size=enc_out_size,
            output_size=187,
            activation="tanh",
            dropout=dropout
        )

    def forward(self, enc_out):
        out = torch.flatten(enc_out, 1, -1)
        out = self.FC_layers(out)

        return out

class DecoderSrcReg(nn.Module):
    """
    Source-specific regressor
    """
    def __init__(
        self,
        enc_out_size,
        Nsound,
        dropout
    ):
        super(DecoderSrcReg, self).__init__()

        self.FC_layers = nn.ModuleList(
            [
                FCModules(
                    input_size=enc_out_size,
                    output_size=2,
                    activation="relu",
                    dropout=dropout
                )
                for _ in range(Nsound)
            ]
        )
        print(f"Number of FC layers: {len(self.FC_layers)}")

    def forward(self, enc_out):
        out_src = torch.flatten(enc_out, 1, -1)

        out = []
        for layers in self.FC_layers:
            out.append(layers(out_src))

        out = torch.cat(out, dim=-1)
        return out

class DecoderEAReg(nn.Module):
    def __init__(
        self,
        enc_out_size,
        Nsound,
        dropout
    ):
        super(DecoderEAReg, self).__init__()
        self.Nsound = Nsound
        
        self.FC_layers_elev = FCModules(
            input_size=enc_out_size,
            output_size=Nsound,
            activation="relu",
            dropout=dropout
        )

        self.FC_layers_azim = FCModules(
            input_size=enc_out_size,
            output_size=2,
            activation="relu",
            dropout=dropout
        )

        self.clamp_elev = nn.Hardtanh(-pi/4, pi/2)
        self.clamp_azim = nn.Hardtanh(0, pi*2)
    def forward(self, enc_out):
        out_elev = torch.flatten(enc_out, 1, -1)
        out_azim = torch.flatten(enc_out, 1, -1)

        out_elev = self.FC_layers_elev(out_elev)
        out_azim = self.FC_layers_azim(out_azim)

        out_elev = self.clamp_elev(out_elev)
        out_azim = self.clamp_azim(out_azim)

        out = []
        for i in range(self.Nsound):
            out.append(out_elev[:, i])
            out.append(out_azim[:, i])
        out = torch.stack(out, dim=-1)

        return out

class TransformerModel(nn.Module):
    def __init__(
        self,
        task,
        Ntime,
        Nfreq,
        Ncues,
        Nsound,
        whichEnc,
        whichDec,
        numEnc=6,
        numFC=3,
        heads=8,
        device="cpu",
        forward_expansion=4,
        dropout=0.1,
        isDebug=False
    ):
        super(TransformerModel, self).__init__()

        self.enc = EncoderDIYTransformer(
            task,
            Ntime,
            Nfreq,
            Ncues,
            Nsound,
            numEnc,
            heads,
            device,
            forward_expansion,
            dropout,
            isDebug
        )

        assert(
            whichDec.lower() in ["ea","src","cls"]
        ), "Invalid decoder structure."

        if whichDec.lower() == "ea":
            self.dec = DecoderEAReg(
                enc_out_size=Nfreq*Ntime*Ncues,
                Nsound=Nsound,
                dropout=dropout
            )
        elif whichDec.lower() == "src":
            self.dec = DecoderSrcReg(
                enc_out_size=Nfreq * Ntime * Ncues,
                Nsound=Nsound,
                dropout=dropout
            )
        elif whichDec.lower() == "cls":
            self.dec = DecoderSrcCls(
                enc_out_size=Nfreq * Ntime * Ncues,
                Nsound=Nsound,
                dropout=dropout
            )

    def forward(self, inputs):
        enc_out = self.enc(inputs)
        out = self.dec(enc_out)
        return out


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)



if __name__ == "__main__":
    path = "./HRTF/IRC*"
    _, locLabel, _ = loadHRIR(path)

    path = "./saved_0508_temp/"
    csvF = pd.read_csv(path+"/train/dataLabels.csv", header=None)


    temp_1 = torch.load(path+"/train/0.pt")
    temp_2 = torch.load(path+"/train/1.pt")
    cues_tensor = torch.stack([temp_1, temp_2], dim=0)
    print(cues_tensor.shape)
    # define
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = "allRegression"
    Nsound = 2
    batch_size = 32
    num_workers = 0
    isPersistent = True if num_workers > 0 else False

    train_dataset = CuesDataset(path + "/train/",
                                task, Nsound, locLabel, isDebug=False)
    train_loader = MultiEpochsDataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=isPersistent
        )
    Nfreq = train_dataset.Nfreq
    Ntime = train_dataset.Ntime
    Ncues = train_dataset.Ncues
    print(f"Nfreq: {Nfreq}, Ntime: {Ntime}, Ncues: {Ncues}")

    model = TransformerModel(
        task=task,
        Ntime=Ntime,
        Nfreq=Nfreq,
        Ncues=Ncues,
        Nsound=Nsound,
        whichEnc="diy",
        whichDec="cls",
        device=device
    )
    model = model.to(device)

    inputs, labels = next(iter(train_loader))
    outputs = model(inputs.to(device))

    print(f"inputs: {inputs.shape},\
        outputs: {outputs.shape}, \
        labels: {labels.shape}")

    # summary(model, (Nfreq, Ntime, Ncues))