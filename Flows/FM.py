import torch 
import numpy as np 
import Dataset.MoG as MoG

""""
we need to construct conditional prob path and velocity field
"""
class ConditionalFM():
    def __init__(self, model, config):
        self.model = model
        self.config = config 
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
    #we'll start with simple linear path aka optimal transport conditional VF
    def compute_sigma(self, t):
        return 1-(self.sigma_max-self.sigma_min)*t
    def compute_mu(self, x ,t):
        return t*x
    def compute_velocity(self, x,x1, t):
        sigma = self.compute_sigma(t)
        return (x1-(self.sigma_max-self.sigma_min)* x) / sigma
    def flow(self, x, x1, t):
        sigma = self.compute_sigma(t)
        mu = self.compute_mu(x1, t)
        return mu + x * sigma
    def sample_from_cond_path(self, x0, x1, t):
        return  self.flow(x0, x1, t)
    def compute_loss(self, x0,x1):
        # sample uniform t, from dataset and from cond prob path and compute loss
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=self.config.device).unsqueeze(1)
        
        # sample from path 
        xt = self.sample_from_cond_path(x0, x1, t)
        score = self.model(xt,t)
        # general form of conditional velocity target
        target=self.compute_velocity(xt, x1, t)
        # in the case of linear interpolation
        #target = (x1-x0) (where sigma min 0 and sigma max 1)
        #mse CFM loss
        loss = torch.mean((score - target)**2)
        return loss
    def get_path(self):
        pass
    def step():
        pass
    def preprocess_batch():
        pass