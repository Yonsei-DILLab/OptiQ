"""Give native DiPo.train the shared dense+NovelD minibatch without changing it."""


class DIPOReplay:
    def __init__(self,replay):self.replay=replay

    def sample(self,batch_size):
        d=self.replay.sample(batch_size)
        # DIPO stores gamma * (1-terminal), whereas the shared replay stores dones.
        return d.observations,d.actions,d.rewards,d.next_observations,.99*(1-d.dones)
