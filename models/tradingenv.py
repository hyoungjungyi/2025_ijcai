import numpy as np
import torch
import torch.nn as nn


def compute_mu_t(w_old, a_target, a0_old,
                 cs, cp,
                 max_iter=50, tol=1e-6):
    """
    Jiang et al. (2017) 식 기반 \mu_t 계산 (iterative).

    Args:
        w_old   : (n,) numpy array, 이전 시점 '위험자산' 비중들 (합<=1)
        a_target: (n,) numpy array, 목표 위험자산 비중들 (합<=1)
        a0_old  : float, 이전 시점 현금 비중 (1 - sum(w_old))
        cs      : float, 매도 수수료율
        cp      : float, 매수 수수료율
        max_iter: int, 최대 반복 횟수
        tol     : float, 수렴 판정 임계값
    Returns:
        mu (float):
            최종 스케일링 팩터. \mu < 1이면 원래 a_target 전부는 못 맞추고,
            mu * a_target 만큼만 위험자산을 보유할 수 있다고 해석.
    """
    mu = 1.0
    for _ in range(max_iter):
        # 매도해야 하는 자산 총량
        sum_sell = 0.0
        for i in range(len(w_old)):
            diff = w_old[i] - mu * a_target[i]
            if diff > 0:
                sum_sell += diff

        # 식의 분자, 분모
        # bracket = 1 - cp*a0_old - (cs + cp - cs*cp)*sum_sell
        # denom   = 1 - cp*a0_old
        bracket = 1.0 - cp * a0_old - (cs + cp - cs * cp) * sum_sell
        denom = 1.0 - cp * a0_old

        if abs(denom) < 1e-12:
            mu_new = 0.0
        else:
            mu_new = bracket / denom

        if abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new

    if mu < 0:
        mu = 0.0
    return mu


class TradingEnvWithMu:
    """
    \mu_t 논리를 사용하는 간단 예시 환경.
    - n개 위험자산 + 현금(무위험자산) 고려 -> 합 1
    - step 시, "목표 위험자산 비중 a_target"이 들어오면,
      \mu_t를 구해 최종 비중 w_new를 결정.
    - reward = 1-step 수익률 (간단 예시)
    """

    def __init__(self, n_assets=3,
                 init_portfolio_value=1_000_000,
                 sell_fee=0.001,  # cs
                 buy_fee=0.001,  # cp
                 device='cpu'):
        self.n_assets = n_assets
        self.portfolio_value = init_portfolio_value

        # 이전 시점 자산 비중: 전부 현금(무위험)
        self.w_old = np.zeros(n_assets)  # (n,)
        self.a0_old = 1.0  # 현금 100%

        # 매도/매수 수수료
        self.cs = sell_fee
        self.cp = buy_fee

        self.device = device

        # 기록용
        self.weights_memory = []
        self.value_memory = []

    def reset(self, init_value=None):
        """
        환경 초기화: 전부 현금 상태로 되돌림.
        """
        if init_value is not None:
            self.portfolio_value = init_value
        else:
            self.portfolio_value = 1_000_000
        self.w_old = np.zeros(self.n_assets)
        self.a0_old = 1.0
        self.weights_memory.clear()
        self.value_memory.clear()
        return (self.w_old, self.a0_old, self.portfolio_value)

    def step(self, a_target, returns):
        """
        a_target : (n,) tensor or np array
                   목표 위험자산 비중(합 <= 1).
        returns  : (n,) 해당 스텝의 자산별 수익률 (예: +0.02 => 2%)

        1) \mu_t 계산 -> 실제 w_new = mu_t * a_target
        2) 수익률 반영하여 portfolio_value 업데이트
        3) reward 계산
        """
        # 텐서->numpy 변환
        if isinstance(a_target, torch.Tensor):
            a_target = a_target.detach().cpu().numpy()
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()

        # 1) \mu_t 계산
        mu = compute_mu_t(
            w_old=self.w_old,
            a_target=a_target,
            a0_old=self.a0_old,
            cs=self.cs,
            cp=self.cp,
            max_iter=100
        )

        # 달성된 위험자산 비중
        w_new = mu * a_target
        sum_wnew = np.sum(w_new)

        # 새 현금 비중
        a0_new = 1.0 - sum_wnew
        if a0_new < 0:
            a0_new = 0.0  # 방어적 처리

        # 2) 포트폴리오 수익:
        #    (위험자산) sum(w_new * returns),
        #    (현금) a0_new는 수익 0 가정
        portfolio_return = np.sum(w_new * returns)

        new_pf_value = self.portfolio_value * (1.0 + portfolio_return)

        # reward = 1-step 수익률
        reward = (new_pf_value - self.portfolio_value) / self.portfolio_value

        # 3) 상태 갱신
        self.portfolio_value = new_pf_value
        self.w_old = w_new
        self.a0_old = a0_new

        self.weights_memory.append((w_new, a0_new))
        self.value_memory.append(new_pf_value)

        return reward, new_pf_value



import numpy as np


class TradingEnvironment:
    def __init__(self,args):

        self.args = args
        self.initial_amount = 1.0
        self.terminal = False
        self.seq_len = self.args.seq_len
        self.fee_rate = self.args.fee_rate
        if args.complex_fee:
            self.cs = self.fee_rate * 2
            self.cp = self.fee_rate
        self.n_assets = self.args.num_stocks
        self.w_old = np.zeros(self.args.num_stocks)
        self.a0_old =  1.0


        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = []
        self.transaction_cost_memory = []
        self.rollout =[]
        self.rollout_len = 30



    def reset(self):
        self.rollout = []
        self.w_old = np.zeros(self.args.num_stocks)
        self.a0_old = 1.0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = []
        # self.date_memory = [self.dataset.data.index.get_level_values('date').unique()[0]]
        self.transaction_cost_memory = []

    def step(self, weights,returns):
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()

        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().numpy()
        if self.args.complex_fee:
            mu = compute_mu_t(
                w_old=self.w_old,
                a_target=weights,
                a0_old=self.a0_old,
                cs=self.cs,
                cp=self.cp,
                max_iter=100)
            w_new = mu * weights
            sum_wnew = np.sum(w_new)
            a0_new = 1.0 - sum_wnew
            if a0_new <0:
                a0_new = 0.0
            portfolio_return = np.sum(w_new * returns)
            new_pf_value = self.portfolio_value * (1 + portfolio_return)
            reward =(new_pf_value - self.portfolio_value) / self.portfolio_value
            self.portfolio_value = new_pf_value
            self.w_old = w_new
            self.a0_old = a0_new
            self.weights_memory.append(( a0_new,w_new))
            self.portfolio_return_memory.append(portfolio_return)
            self.asset_memory.append(new_pf_value)
            return reward
        else:
            self.weights_memory.append(weights)
            portfolio_return = np.sum(weights * returns)
            change_ratio = returns + 1
            weights_brandnew = self.normalization(weights * change_ratio)
            self.weights_memory.append(weights_brandnew)
            weights_old = (self.weights_memory[-3])
            weights_new = (self.weights_memory[-2])
            diff_weights = np.sum(np.abs(weights_old - weights_new), axis=-1)
            transcationfee = diff_weights * self.fee_rate * self.portfolio_value
            new_portfolio_value = (self.portfolio_value -transcationfee) * (1 + portfolio_return)
            portfolio_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
            reward = portfolio_return

            ###reward
            # entropy_value = -np.sum(weights * np.log(weights + 1e-6))
            # reward = portfolio_return
            # ##entropy_reward
            # entropy_penalty_coeff = 1  # This coefficient needs to be tuned based on the specific problem
            # penalty = entropy_value * entropy_penalty_coeff
            # reward -= penalty

            self.portfolio_value = new_portfolio_value
            self.portfolio_return_memory.append(portfolio_return)
            # self.date_memory.append(date[0])
            self.asset_memory.append(new_portfolio_value)

            return reward

    def normalization(self, actions):
        # a normalization function not only for actions to transfer into weights but also for the weights of the
        # portfolios whose prices have been changed through time
        sum = np.sum(actions, axis=-1, keepdims=True)
        actions = actions / sum
        return actions