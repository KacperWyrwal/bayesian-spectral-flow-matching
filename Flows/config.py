import torch 

class FlowConfig():
    def __init__(
        self,
        num_steps=100,
        lr=1e-4,
        sigma_min=0.01,
        sigma_max=1.0,
        use_spectral=False,
        use_sobolev_ot=False,
        use_matern_source=False,
        alpha=1.0,
        beta=1.0,
        nu=2.0,
        tau=1.0,
        laplacian=None,
        spectral_loss_modes=None,
        spectral_loss_basis="laplacian",
        grid_shape=None,
    ):
        self.num_steps = num_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_spectral = use_spectral
        self.use_sobolev_ot = use_sobolev_ot
        self.use_matern_source = use_matern_source
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.tau = tau
        self.laplacian = laplacian
        self.spectral_loss_modes = spectral_loss_modes
        self.spectral_loss_basis = spectral_loss_basis
        self.grid_shape = grid_shape

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
    def __init__(
        self,
        input_dim,
        output_dim,
        intermediate_dim,
        enable_time=True,
        model_type="mlp",
        grid_shape=None,
        fno_modes=16,
        fno_modes2=None,
        fno_width=64,
        fno_layers=4,
        fno_proj_dim=None,
        fno_use_norm=True,
        laplacian=None,
        spectral_modes=None,
        sobolev_alpha=0.0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim
        self.enable_time = enable_time
        self.model_type = model_type
        self.grid_shape = grid_shape
        self.fno_modes = fno_modes
        self.fno_modes2 = fno_modes2
        self.fno_width = fno_width
        self.fno_layers = fno_layers
        self.fno_proj_dim = fno_proj_dim
        self.fno_use_norm = fno_use_norm
        self.laplacian = laplacian
        self.spectral_modes = spectral_modes
        self.sobolev_alpha = sobolev_alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        