"""This file contains different model architectures
The structure of the file and the fomder is temporary and used for dummy tests rights now 
we can split into other files later
"""

import torch 

class dummy_mlp(torch.nn.Module):
    def __init__(self, model_config):
        super(dummy_mlp, self).__init__()
        self.model_config = model_config
        # do not mutate the passed config object; compute internal input dim
        in_dim = model_config.input_dim + (1 if model_config.enable_time else 0)
        self.linear1 = torch.nn.Linear(in_dim, 2*model_config.intermediate_dim)
        self.linear2 = torch.nn.Linear(2*model_config.intermediate_dim, model_config.intermediate_dim)
        self.linear3 = torch.nn.Linear(model_config.intermediate_dim, model_config.intermediate_dim)
        self.linear4 = torch.nn.Linear(model_config.intermediate_dim, 2*model_config.intermediate_dim)
        self.linear5 = torch.nn.Linear(2*model_config.intermediate_dim, model_config.output_dim)

    def forward(self, x, t=None):
        if t is not None and self.model_config.enable_time:
            x = torch.cat([x, t], dim=1)
        x = self.linear1(x)
        x = torch.selu(x)
        x = self.linear2(x)
        x = torch.selu(x)
        x = self.linear3(x)
        x = torch.selu(x)
        x = self.linear4(x)
        x = torch.selu(x)
        x = self.linear5(x)
        return x