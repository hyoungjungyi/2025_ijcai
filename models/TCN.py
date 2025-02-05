import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import TemporalAttention

class TCNBlock(nn.Module):
    """
    A single residual block for TCN with dilated convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual Connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = None

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + res  # Residual Connection 추가

class TCN(nn.Module):
    """
    Temporal Convolutional Network with residual blocks.
    """
    def __init__(self, input_dim, output_dim, num_layers, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            layers.append(TCNBlock(input_dim if i == 0 else output_dim,
                                   output_dim, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Model(nn.Module):
    """
    TCN-based model for time series prediction with Temporal Attention.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # TCN Encoder
        self.encoder_tcn = TCN(configs.d_model, configs.d_model, configs.e_layers, kernel_size=3, dropout=configs.dropout)

        # TCN Decoder
        self.decoder_tcn = TCN(configs.d_model, configs.d_model, configs.d_layers, kernel_size=3, dropout=configs.dropout)

        # Temporal Attention
        self.Temporal = TemporalAttention(configs.d_model)

        # Output Projection
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [Batch, Seq, Feature]
        enc_out = enc_out.permute(0, 2, 1)  # [Batch, Feature, Seq] for TCN
        enc_out = self.encoder_tcn(enc_out)  # [Batch, Feature, Seq]
        enc_out = enc_out.permute(0, 2, 1)  # [Batch, Seq, Feature]

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [Batch, Seq, Feature]
        dec_out = dec_out.permute(0, 2, 1)  # [Batch, Feature, Seq] for TCN
        dec_out = self.decoder_tcn(dec_out)  # [Batch, Feature, Seq]
        dec_out = dec_out.permute(0, 2, 1)  # [Batch, Seq, Feature]

        # Temporal Attention 적용
        if self.configs.moe_train:
            dec_out = self.projection(dec_out[:, -self.pred_len:, :]).squeeze(-1)
        else:
            dec_out = self.Temporal(dec_out)  # [Batch, Seq, Feature] → [Batch, 1, Feature]
            dec_out = self.projection(dec_out)  # [Batch, 1, c_out]

        if self.output_attention:
            return dec_out, None  # TCN doesn't have attention maps
        else:
            return dec_out