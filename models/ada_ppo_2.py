import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from models import Transformer, Informer, Reformer, Autoformer, Fedformer, Flowformer, Flashformer, itransformer, crossformer, deformableTST
from layers.SelfAttention_Family import TemporalAttention
import os

class ADA_PPO(nn.Module):
    def __init__(self, model_name, configs, setting, deterministic=False):
        super(ADA_PPO, self).__init__()
        self.model_name = configs.model
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'Autoformer': Autoformer,
            'Fedformer': Fedformer,
            'itransformer': itransformer,
            'crossformer': crossformer,
            'deformableTST': deformableTST
        }
        self.model = self.model_dict[model_name].Model(configs)

        # Transfer learning 관련 설정
        if configs.transfer:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            if configs.freeze:
                for param in self.model.parameters():
                    param.requires_grad = False

        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.periods = configs.horizons
        self.num_periods = len(self.periods)
        self.deterministic = deterministic
        self.Temporal = TemporalAttention(configs.d_model)

        self.shared_rep = nn.Sequential(
            nn.Linear(configs.d_model, 128),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )

        self.horizon_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(128, self.num_periods)
        )
        self.portfolio_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(128, self.pred_len)
        )

        # 각 거래 기간별 포트폴리오 할당을 위한 Actor-Critic 레이어들
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
        # 포트폴리오 예측을 위한 head (필요에 따라 사용)
        self.layer_pred = nn.Linear(self.pred_len, 1)

    def pi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Transformer 기반 백본으로부터 예측 결과 산출
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # shape: [N, pred_len]
        pred_scores =self.Temporal(pred_scores)
        shared_features = self.shared_rep(pred_scores)
        horizon_logits = self.horizon_head(shared_features)  # shape: [N, num_periods]

        epsilon = 1e-8
        selection_probs = F.softmax(horizon_logits, dim=1) + epsilon
        selection_probs = selection_probs / selection_probs.sum(dim=1, keepdim=True)
        selection_dist = Categorical(selection_probs)
        selection_entropy = selection_dist.entropy()

        if (not self.training) or self.deterministic:
            # 평가 시 (deterministic): 가장 높은 확률을 가진 거래 기간 선택
            selected_period_indices = torch.argmax(selection_probs, dim=1)  # [N]
        else:
            # 학습 시: 확률 분포에서 샘플링
            selected_period_indices = selection_dist.sample()  # [N]

        # --- 포트폴리오 할당을 위한 분리된 처리 ---
        # 포트폴리오 할당 head를 통해 별도의 표현 산출
        portfolio_repr = self.portfolio_head(shared_features)  # shape: [N, pred_len]

        N = pred_scores.shape[0]
        actions = torch.zeros(N, device=pred_scores.device)  # 최종 action을 저장할 변수
        final_log_prob = torch.zeros(N, device=pred_scores.device)
        action_entropy_buff = torch.zeros(N, device=pred_scores.device)

        # Horizon 선택의 log_prob 계산
        selection_log_prob = selection_dist.log_prob(selected_period_indices)

        # 각 horizon 별로 포트폴리오 할당 분포 계산
        for i, period in enumerate(self.periods):
            mask = (selected_period_indices == i)  # [N] boolean mask
            M = mask.sum()
            if M == 0:
                continue
            # portfolio_repr에서 해당하는 샘플만 선택하여 사용
            selected_portfolio_repr = portfolio_repr[mask, :]
            mu = self.layer_mu[i](selected_portfolio_repr).squeeze(-1)  # [M]
            std = torch.clamp(F.softplus(self.layer_std[i](selected_portfolio_repr)), min=1e-2).squeeze(-1)  # [M]

            dist = Normal(mu, std)
            if (not self.training) or self.deterministic:
                # 평가 시: action = tanh(mu)
                action = torch.tanh(mu).squeeze(-1)
                real_log_prob = torch.zeros_like(action)
                action_entropy = torch.zeros_like(action)
            else:
                # 학습 시: 샘플링 및 로그 보정 적용
                action_sample = dist.rsample()  # reparameterization trick
                action = torch.tanh(action_sample)  # [M]
                log_prob = dist.log_prob(action_sample)
                log_prob_tanh_correction = -torch.log(torch.clamp(1 - action.pow(2), min=1e-5))
                real_log_prob = log_prob + log_prob_tanh_correction
                action_entropy = dist.entropy()

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
        # 추가: 선택된 horizon 중 다수결 등 후처리(필요 시) 적용 가능

        return actions, total_log_prob, total_entropy, final_idx

    def value(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred_scores = self.Temporal(pred_scores)
        shared_features = self.shared_rep(pred_scores)
        portfolio_repr = self.portfolio_head(shared_features)
        horizon_logits = self.horizon_head(shared_features)
        selection_probs = F.softmax(horizon_logits, dim=1)  # [N, num_periods]

        # 각 horizon 별로 value를 산출한 후, 가중합
        value_list = []
        for i, period in enumerate(self.periods):
            value_i = self.layer_value[i](portfolio_repr).squeeze(-1)  # [N]
            value_list.append(value_i)
        values_all = torch.stack(value_list, dim=1)  # [N, num_periods]
        value_portfolio = (selection_probs * values_all).sum(dim=1)  # [N]
        return value_portfolio.mean(dim=0).squeeze()  # 스칼라 반환

    def pred(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 포트폴리오 할당 head를 통해 최종 예측 산출
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred_scores = self.Temporal(pred_scores)
        shared_features = self.shared_rep(pred_scores)
        portfolio_repr = self.portfolio_head(shared_features)
        pred_return = self.layer_pred(portfolio_repr).squeeze(-1)
        return pred_return
