import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal,Normal
from models import Transformer,Informer,Reformer,Autoformer,Fedformer,Flowformer,Flashformer,itransformer,crossformer

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super(TemporalAttention, self).__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)  # Last time step as query
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D], [N, D, 1] -> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # Attention weights
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] -> [N, D]
        return output

class PPO(nn.Module):
    def __init__(self, model_name,configs):
        super(PPO, self).__init__()
        self.model_name = configs.model
        # self.model_dict ={'Transformer': Transformer}
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

        # Initialize model and parameters
        self.model = self.model_dict[model_name].Model(configs)
        self.d_model = configs.d_model
        # self.num_stocks = configs.num_stocks
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        # Actor-Critic Layers
        if self.c_out > 1:
            self.layer_mu = nn.Linear(self.c_out, 1)
            self.layer_std = nn.Linear(self.c_out, 1)
            # self.layer_pred = nn.Linear(self.c_out, self.num_stocks)
            self.layer_value_1 = nn.Linear(self.c_out, 128)
            self.layer_value_2 = nn.Linear(128, 1)

        else:
            self.layer_mu = None
            self.layer_std = None
            self.layer_pred = None
            self.layer_value_1 = nn.Linear(1, 128)
            self.layer_value_2 = nn.Linear(128, 1)

    def pi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Forward pass through the model
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Compute mean (mu) and standard deviation (std)
        if self.c_out > 1:
            mu = self.layer_mu(pred_scores).squeeze(-1)
            std = torch.clamp(F.softplus(self.layer_std(pred_scores)), min=1e-2).squeeze(-1)
        else:
            mu = pred_scores.squeeze(-1)  # Directly use pred_scores as mu
            std = torch.clamp(F.softplus(pred_scores), min=1e-2).squeeze(-1)# Ensure std is positive

        # If not training, return deterministic action
        if not self.training:
            real_action = torch.tanh(mu)
            real_action = real_action
            return real_action.squeeze(-1), None

        # Otherwise, sample from the distribution
        # cov_matrix = torch.diag_embed(std)
        # dist = MultivariateNormal(mu, cov_matrix)
        dist = Normal(mu, std)

        action = dist.rsample()
        real_action = torch.tanh(action)

        # Calculate log probabilities
        log_prob = dist.log_prob(action)
        real_log_prob = log_prob - torch.log(torch.clamp(1 - real_action.pow(2), min=1e-5))


        return real_action, real_log_prob.sum()

    def value(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Compute value
        # if self.c_out > 1:
        #     intermediate = F.relu(self.layer_value_1(pred_scores)) #
        # else:
        #     intermediate = F.relu(self.layer_value_1(pred_scores))
        if self.c_out > 1:
            intermediate = self.layer_value_1(pred_scores)#
        else:
            intermediate = self.layer_value_1(pred_scores)

        value = self.layer_value_2(intermediate)
        value_portfolio = value.mean(dim=0)

        return value_portfolio.squeeze()
    def pred(self,x_enc,x_mark_enc,x_dec,x_mark_dec):

        pred_scores = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return pred_scores.squeeze(-1)

# Example usage
if __name__ == '__main__':
    # Dummy input data
    class Config:
        model = 'Transformer'
        d_model = 512
        num_stocks = 30
        pred_len = 5
        c_out = 16
        attention_bool = False

    configs = Config()

    batch_size = 16
    num_stocks = configs.num_stocks
    seq_len = 10
    num_features = 5

    x_enc = torch.randn((batch_size, num_stocks, seq_len, num_features))
    x_mark_enc = torch.randn((batch_size, num_stocks, seq_len, num_features))
    x_dec = torch.randn((batch_size, num_stocks, seq_len, num_features))
    x_mark_dec = torch.randn((batch_size, num_stocks, seq_len, num_features))

    model = PPO(configs)
    action, log_prob = model.pi(x_enc, x_mark_enc, x_dec, x_mark_dec)
    value = model.value(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print("Action:", action)
    print("Log Probability:", log_prob)
    print("Value:", value)
