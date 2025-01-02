import numpy as np


class TradingEnvironment:
    def __init__(self,args):

        self.args = args
        self.initial_amount = 1.0
        self.terminal = False
        self.seq_len = self.args.seq_len
        self.fee_rate = self.args.fee_rate
        self.num_stocks = self.args.num_stocks


        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [np.array([1 / (self.num_stocks)] * (self.num_stocks))]
        # self.date_memory = [self.dataset.data.index.get_level_values('date').unique()[0]]
        self.transaction_cost_memory = []
        self.rollout =[]
        self.rollout_len = 20
        self.local_loss = []



    def reset(self):
        self.rollout = []
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [np.array([1 / (self.num_stocks)] * (self.num_stocks))]
        # self.date_memory = [self.dataset.data.index.get_level_values('date').unique()[0]]
        self.transaction_cost_memory = []

    def step(self, weights,returns):
         # make judgement about whether our data is running ou
        weights = np.array(weights.detach().cpu())
        returns = np.array(returns.detach().cpu())
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