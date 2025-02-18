import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from models import Transformer, Informer, Reformer, Autoformer, Fedformer, Flowformer, Flashformer, itransformer, crossformer
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
            'crossformer': crossformer
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
        self.layer_mu = nn.Linear(self.pred_len, 1)
        self.layer_std = nn.Linear(self.pred_len, 1)
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
        horizon_entropy = selection_dist.entropy()


        if (not self.training) or self.deterministic:
            selected_horizon = torch.argmax(selection_probs, dim=1)  # [N]
        else:
            selected_horizon = selection_dist.sample()
        unique_vals, counts = selected_horizon.unique(return_counts=True)
        final_idx = unique_vals[counts.argmax()]
            # --- 포트폴리오 할당을 위한 분리된 처리 ---
        # 포트폴리오 할당 head를 통해 별도의 표현 산출
        portfolio_repr = self.portfolio_head(shared_features)  # shape: [N, pred_len]
        mu = self.layer_mu(portfolio_repr).squeeze(-1)
        std = torch.clamp(F.softplus(self.layer_std(portfolio_repr)), min=1e-2).squeeze(-1)
        # If not training, return deterministic action
        if (not self.training) or self.deterministic:
            action = torch.tanh(mu)
            return action, None, horizon_entropy.mean(), final_idx

        action_dist = Normal(mu, std)
        raw_action = action_dist.rsample()  # [N]
        action = torch.tanh(raw_action)

        log_prob = action_dist.log_prob(raw_action)
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-5))
        log_prob = log_prob.squeeze(-1)

        horizon_log_prob = selection_dist.log_prob(selected_horizon)  # [N]
        total_log_prob = horizon_log_prob + log_prob
        action_entropy = action_dist.entropy()  # [N]
        total_entropy = (horizon_entropy + action_entropy).mean()


        return action, total_log_prob.sum(), total_entropy, final_idx

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
