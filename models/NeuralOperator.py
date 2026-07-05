import torch
import torch.nn as nn
# whe neural operator : we need to have point wise ,
# mesh free , predict in function space , 
# why though , maybe the neural operator paradigm fits velocity better n it should in theory , it is made for actual functionthat we should be able to derivate and integrate
# for neural operator we need to integrate 

class NeuralOperator(nn.Module):
    def __init__(self,config):
        super(NeuralOperator, self).__init__()
        self.config = config
        self.basis = None
        self.down_project = None
        self.up_project = None
        self.KernelLayers =nn.ModuleList([KernelLayer(config) for _ in range(self.config.num_kernel_layers)])
    def KernalIntegrator():
        pass
    def forward(self,x):
        for layer in self.KernelLayers:
            x = layer(x)
        return x
    
class KernelIntegrator(nn.Module):
    def __init__(self,config):
        super(KernelIntegrator, self).__init__()
        self.config = config
    def integrate(self,x):
        integrand=torch.sum()
        return integrand
    
class KernelLayer(nn.Module):
    def __init__(self,config):
        super(KernelLayer, self).__init__()
        self.config = config
        self.integrator = KernelIntegrator(config)
        self.linear = nn.Linear(config.input_dim, config.output_dim)
        self.activation = nn.ReLU()
    def forward(self,x):
        x_integrated=self.integrator.integrate(x)
        x=self.linear(x)
        x=self.activation(x+x_integrated)
        return x
    