import torchdiffeq
import torch

class ODEFunc(torch.nn.Module):
    def __init__(self, model):
        super(ODEFunc, self).__init__()
        self.model = model

    def forward(self, t, x):
        # `t` is a scalar 0-d tensor provided by torchdiffeq; expand it
        # to a column vector of shape [batch, 1] to match `x`.
        batch = x.shape[0]
        t_col = t.unsqueeze(0).expand(batch, 1).to(x.dtype).to(x.device)
        return self.model(x, t_col)

def infer(model, x, config):
    # define the ODE function for the flow
    ode_func = ODEFunc(model)
    # define the time points for integration
    t = torch.linspace(0, 1, steps=config.num_steps).to(config.device)
    # integrate the ODE to get the flow output
    flow_output = torchdiffeq.odeint(ode_func, x, t)
    return flow_output
    