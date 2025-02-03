import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers.cross_encoder import Encoder
from layers.cross_decoder import Decoder
from layers.SelfAttention_Family import TemporalAttention
from layers.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from layers.cross_embed import DSW_embedding

from math import ceil

class  Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        # 시계열 관련 설정
        self.data_dim = configs.enc_in
        self.in_len = configs.seq_len
        self.out_len = configs.pred_len
        self.seg_len = 6
        self.merge_win = 2
        self.baseline = getattr(configs, 'baseline', False)
        self.pad_in_len = ceil(self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        self.enc_value_embedding = DSW_embedding(self.seg_len, configs.d_model)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), configs.d_model)
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)

        self.encoder = Encoder(
            e_blocks=configs.e_layers,
            win_size=self.merge_win,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            block_depth=1,
            dropout=configs.dropout,
            in_seg_num=(self.pad_in_len // self.seg_len),
            factor=configs.factor
        )

        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), configs.d_model)
        )
        self.decoder = Decoder(
            seg_len=self.seg_len,
            d_layers=configs.e_layers + 1,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            out_seg_num=(self.pad_out_len // self.seg_len),
            factor=configs.factor
        )

        self.Temporal = TemporalAttention(configs.dec_in)
        self.projection = nn.Linear(configs.dec_in, configs.c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_seq = x_enc  # Crossformer는 주로 하나의 입력 x_seq 사용

        # (1) baseline 보정값 계산
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)  # 예시
        else:
            base = 0

        # (2) 필요 시 앞부분 padding (Crossformer 내부 seg_len에 맞추기 위해)
        if self.in_len_add != 0:
            # 원 코드 예시처럼 "x_seq[:, :1, :].expand" 로 반복 padding
            pad_part = x_seq[:, :1, :].expand(-1, self.in_len_add, -1)
            x_seq = torch.cat((pad_part, x_seq), dim=1)

        # (3) 인코더 임베딩 (Value Embedding + Pos Embedding)
        # x_seq shape: [Batch, pad_in_len, data_dim]가 되어야 하며,
        # Crossformer 코드에서 [Batch, data_dim, pad_in_len//seg_len, d_model] 형태로 재배치할 수도 있음
        x_seq = self.enc_value_embedding(x_seq)
        x_seq = x_seq + self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        # (4) Encoder 수행
        enc_out = self.encoder(x_seq)

        # (5) Decoder 입력 구성
        #     dec_pos_embedding 을 batch_size만큼 확장
        batch_size = x_enc.shape[0]
        dec_in = repeat(self.dec_pos_embedding, 'b d l dm -> (repeat b) d l dm', repeat=batch_size)

        # (6) Decoder 수행
        dec_out = self.decoder(dec_in, enc_out)
        # dec_out shape: [Batch, pad_out_len, data_dim] (원 코드상에서 transpose 등 형태 변화 가능)

        # padding이 들어갔으면 실제 pred_len만큼 slice
        # 원 코드 예시대로라면 dec_out[:, :self.out_len, :]
        dec_out = base + dec_out[:, :self.out_len, :]

        # (7) proj/TemporalAttention 등 최종 투영
        #     Vanilla Transformer 예시처럼 moe_train 여부로 분기
        if self.configs.moe_train:
            # 예: [B, out_len, d_model] -> [B, pred_len], 여기서는 마지막 차원 c_out
            dec_out = self.projection(dec_out[:, -self.out_len:, :]).squeeze(-1)
        else:
            dec_out = self.Temporal(dec_out)  # [B, out_len, d_model]
            dec_out = self.projection(dec_out)  # [B, out_len, c_out]

        return dec_out

