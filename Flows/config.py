import torch 

class FlowConfig():
    def __init__(self, num_steps=100, lr=1e-4,sigma_min=0.01, sigma_max=1.0):
        self.num_steps = num_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

class SDEConfig():
    def __init__(self, num_steps=1000):
        self.num_steps = num_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 1e-4
class ODEConfig():
    def __init__(self, num_steps=1000):
        self.num_steps = num_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 1e-4
class ModelConfig():
    def __init__(self, input_dim, output_dim, intermediate_dim, enable_time=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim
        self.enable_time = enable_time
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        