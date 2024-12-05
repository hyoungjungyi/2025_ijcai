from exp.exp_basic import Exp_Basic
class Exp_Reinforce(Exp_Basic):
    def __init__(self,config):
        super(Exp_Reinforce,self).__init__(config)
        # self.agent = REINFORCE(self.config)
        # self.buffer = ReplayBuffer(self.config.buffer_limit)
        # self.agent.load_model()
        # self.agent.model.eval()