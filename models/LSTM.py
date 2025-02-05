import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import TemporalAttention

class Model(nn.Module):
    """
    LSTM-based model for time series prediction with sequential processing.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # LSTM Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout,
            batch_first=True
        )

        # LSTM Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=configs.d_layers,
            dropout=configs.dropout,
            batch_first=True
        )

        # Temporal Attention Layer (optional, for compatibility with Transformer architecture)
        self.Temporal = TemporalAttention(configs.d_model)

        # Output Projection
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, (h_n, c_n) = self.encoder_lstm(enc_out)

        #디코더 레이어만큼만 가져오기
        h_n = h_n[:self.decoder_lstm.num_layers]
        c_n = c_n[:self.decoder_lstm.num_layers]

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out, _ = self.decoder_lstm(dec_out, (h_n,c_n))

        # moe 아닐 경우 temporal attention 적용
        if self.configs.moe_train:
            dec_out = self.projection(dec_out[:, -self.pred_len:, :]).squeeze(-1)
        else:
            dec_out = self.Temporal(dec_out)
            dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out, None  # LSTM doesn't have attention maps
        else:
            return dec_out