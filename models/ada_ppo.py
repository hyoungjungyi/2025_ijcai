import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from models import Transformer  # 올바른 import 경로를 확인하세요

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super(TemporalAttention, self).__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)  # 마지막 시간 스텝을 query로 사용
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # 어텐션 가중치
        output = torch.matmul(lam, z).squeeze(1)  # [N, D]
        return output

class ADA_PPO(nn.Module):
    def __init__(self, model_name, configs, deterministic=False):
        super(ADA_PPO, self).__init__()
        self.model_name = configs.model
        self.model_dict = {'Transformer': Transformer}


        # 모델과 파라미터 초기화
        self.model = self.model_dict[model_name].Model(configs)
        self.d_model = configs.d_model

        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.periods = configs.horizons
        self.num_periods = len(self.periods)
        self.deterministic = deterministic

        # 각 거래 기간에 대한 Actor-Critic 레이어
        self.layer_mu = nn.ModuleList([
            nn.Linear(self.pred_len, 1) for _ in self.periods
        ])
        self.layer_std = nn.ModuleList([
            nn.Linear(self.pred_len, 1) for _ in self.periods
        ])
        self.layer_value = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.pred_len, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in self.periods
        ])

        self.layer_pred = nn.Linear(self.pred_len, 1)
        # 거래 기간 선택을 위한 선택 레이어
        self.selection_layer = nn.Linear(self.pred_len, self.num_periods)

    def pi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 모델을 통한 순전파
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [N, P]

        N = pred_scores.shape[0]

        # 선택 로짓과 확률
        selection_logits = self.selection_layer(pred_scores)  # [N, num_periods]
        selection_probs = F.softmax(selection_logits, dim=1)  # [N, num_periods]
        selection_dist = Categorical(selection_probs)
        selection_entropy = selection_dist.entropy()


        if self.deterministic:
            # 그리디 선택: 가장 높은 확률을 가진 거래 기간 선택
            selected_period_indices = torch.argmax(selection_probs, dim=1)  # [N]
        else:
            # 샘플링: 확률 분포에서 거래 기간 샘플링
            selected_period_indices = selection_dist.sample()  # [N]



        # 3) horizon별로 action을 구하기 위한 변수
        actions = torch.zeros(N, device=pred_scores.device)  # [N]
        final_log_prob = torch.zeros(N, device=pred_scores.device)
        action_entropy_buff = torch.zeros(N, device=pred_scores.device)

        # horizon 선택의 log_prob
        selection_log_prob = selection_dist.log_prob(selected_period_indices)

        for i, period in enumerate(self.periods):
            mask = (selected_period_indices == i)  # [N], True/False
            M = mask.sum()
            if M == 0:
                continue

            # 해당 horizon(i)을 선택한 샘플만 골라옴
            # selected_pred_scores = shape [M, period]
            # 여기서는 "마지막 period개"를 뽑는다고 가정(e.g. [-5:] -> 15~19)
            selected_pred_scores = pred_scores[mask, :period]

            # layer_mu[i], layer_std[i]는 in_features=self.pred_len이므로,
            # pad해서 [M, pred_len]으로 맞춤
            # pad: (left, right) => period -> self.pred_len
            pad_len = self.pred_len - period  # how many columns to add on the left
            # F.pad(input, pad=(left, right), ...)
            selected_pred_scores_pad = F.pad(selected_pred_scores, (0, pad_len), "constant", 0)
            # now shape=[M, self.pred_len]

            mu = self.layer_mu[i](selected_pred_scores_pad).squeeze(-1)  # [M]
            std = torch.clamp(F.softplus(self.layer_std[i](selected_pred_scores_pad)), min=1e-2).squeeze(-1)  # [M]

            dist = Normal(mu, std)
            if not self.training and self.deterministic:
                # 평가 시(deterministic) -> action = tanh(mu)
                action = torch.tanh(mu).squeeze(-1)
                real_log_prob = torch.zeros_like(action)
                action_entropy = torch.zeros_like(action)
            else:
                # 확률적

                action_sample = dist.rsample()  # [M]
                action = torch.tanh(action_sample)  # [M]

                log_prob = dist.log_prob(action_sample)  # [M]
                # tanh에 대한 log_prob 보정
                log_prob_tanh_correction = -torch.log(
                    torch.clamp(1 - action.pow(2), min=1e-5)
                )
                real_log_prob = log_prob + log_prob_tanh_correction
                action_entropy = dist.entropy()

            # actions, log_probs_tensor에 할당
            actions[mask] = action
            action_entropy_buff[mask] = action_entropy
            final_log_prob[mask] = selection_log_prob[mask] + real_log_prob

        vals, counts = selected_period_indices.unique(return_counts=True)
        max_count = counts.max()
        tie_mask = (counts == max_count)
        tie_indices = vals[tie_mask]
        final_idx = tie_indices.min()
        total_entropy = (selection_entropy + action_entropy_buff).mean()
        total_log_prob = final_log_prob.sum()

        return actions, total_log_prob, total_entropy, final_idx

    def value(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [N, pred_len]
        # value = self.layer_value_1(pred_scores)
        # value_portfolio= self.layer_value_2(value).mean(dim=0)

        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [N, pred_len]
        N = pred_scores.shape[0]

        # 1) 현재 state에서 horizon 선택 확률
        selection_logits = self.selection_layer(pred_scores)  # [N, num_periods]
        selection_probs = F.softmax(selection_logits, dim=1)  # [N, num_periods]

        # 2) 각 horizon별 value_i를 구한 뒤, selection_probs로 가중합
        value_list = []
        for i, period in enumerate(self.periods):
            sub_scores = pred_scores[:, :period]  # 마지막 period개
            pad_len = self.pred_len - period
            sub_scores_pad = F.pad(sub_scores, (0, pad_len), "constant", 0)  # [N, pred_len]

            value_i = self.layer_value[i](sub_scores_pad).squeeze(-1)  # [N]
            value_list.append(value_i)

        # [N, num_periods]
        values_all = torch.stack(value_list, dim=1)

        # 가중합 -> shape: [N]
        # PPO의 Critic은 "현재 상태 s에서의 V^\pi(s)"를 추정하므로
        # horizon i가 선택될 확률 * 그 i일 때의 가치 -> 합
        value_portfolio = (selection_probs * values_all).sum(dim=1)  # [N]

        return value_portfolio.mean(dim=0).squeeze() # [1]

    def pred(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred_return = self.layer_pred(pred_scores).squeeze(-1)
        return pred_return