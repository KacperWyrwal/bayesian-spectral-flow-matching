import os, sys; 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import matplotlib.pyplot as plt
import numpy as np
import Flows.FM as FM
import Flows.config as config
import Flows.matern_sampler as matern_sampler
import models.architecture as MA
import Flows.infer as infer
import Flows.logprob as logprob
import Dataset.MoG as MoG
import OT.sampler as ot_sampler


class Trainer:
    """Trainer class for Flow Matching models."""

    @staticmethod
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def __init__(
        self, 
        flow_config=None, 
        model_config=None, 
        num_epochs=100,
        dataset_size=50000,
        batch_size=256,
        checkpoint_path="./checkpoints/cfm_checkpoint.pth",
        source_sampler=None,
        target_sampler=None
    ):
        """
        Initialize the trainer.
        
        Args:
            flow_config: FlowConfig instance with training parameters
            model_config: ModelConfig instance with model parameters
            num_epochs: Number of training epochs
            dataset_size: Size of dataset to generate
            batch_size: Batch size for training
            checkpoint_path: Path to save model checkpoint
            source_sampler: Callable(num_samples, input_dim, device) -> tensor for x0 distribution
                           Default: standard Gaussian
            target_sampler: Callable(num_samples, output_dim, device) -> tensor for x1 distribution
                           Default: Mixture of Gaussians
        """
        self.flow_config = flow_config or config.FlowConfig(
            num_steps=5, lr=5e-5, sigma_min=0.001, sigma_max=1.0
        )
        self.model_config = model_config or config.ModelConfig(
            input_dim=1, output_dim=1, intermediate_dim=256
        )
        self.num_epochs = num_epochs
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        
        # Set default samplers
        self.source_sampler = source_sampler or self._default_source_sampler
        self.target_sampler = target_sampler or self._default_target_sampler
        
        print(f"Device: {self.flow_config.device}")
        
        # Initialize model and flow
        self.model = MA.build_model(self.model_config).to(self.flow_config.device)
        model_type = getattr(self.model_config, "model_type", "mlp")
        total_params, trainable_params = self.count_parameters(self.model)
        print(
            f"Model ({model_type}): {total_params:,} parameters "
            f"({trainable_params:,} trainable)"
        )
        if self.flow_config.use_spectral:
            self.cfm = FM.SpectralFM(self.model, self.flow_config)
        else:
            self.cfm = FM.ConditionalFM(self.model, self.flow_config)

        self.ot_matcher = None
        if self.flow_config.use_spectral and self.flow_config.use_sobolev_ot:
            self.ot_matcher = ot_sampler.OTDisplacementMap(
                self.flow_config,
                eigenvalues=self.cfm._eigenvalues,
                eigenvectors=self.cfm._eigenvectors,
            )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.flow_config.lr
        )
        self.dataloader = None
        self.training_losses = []
    
    def _default_source_sampler(self, num_samples, input_dim, device):
        """Default source distribution: standard Gaussian."""
        if self.flow_config.use_spectral and self.flow_config.use_matern_source:
            if self.flow_config.laplacian is None:
                raise ValueError("FlowConfig.laplacian is required for Matérn source sampling.")
            return matern_sampler.sample_matern_graph(
                num_samples=num_samples,
                laplacian=self.flow_config.laplacian,
                nu=self.flow_config.nu,
                tau=self.flow_config.tau,
                device=device,
            )
        return torch.randn(num_samples, input_dim, device=device)
    
    def _default_target_sampler(self, num_samples, output_dim, device):
        """Default target distribution: Mixture of Gaussians."""
        # output dim here is not used because MoG sampler generates 1D samples by default; an extension for d-dim sampler is possible but not of an interest for now.
        samples = MoG.sample_mog(num_samples).to(device)
        if samples.dim() == 1:
            samples = samples.unsqueeze(1)
        return samples
    
    def prepare_data(self):
        """Prepare training data using configured samplers."""
        x0_all = self.source_sampler(
            self.dataset_size,
            self.model_config.input_dim,
            self.flow_config.device
        )
        x1_all = self.target_sampler(
            self.dataset_size,
            self.model_config.output_dim,
            self.flow_config.device
        )
        dataset = torch.utils.data.TensorDataset(x0_all, x1_all)
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
    
    def train(self):
        """Train the model."""
        if self.dataloader is None:
            self.prepare_data()
        
        for epoch in range(self.num_epochs):
            for batch_idx, (batch_x0, batch_x1) in enumerate(self.dataloader):
                batch_x0 = batch_x0.to(self.flow_config.device)
                batch_x1 = batch_x1.to(self.flow_config.device)

                if self.ot_matcher is not None:
                    batch_x0, batch_x1 = self.ot_matcher.pair_minibatch(batch_x0, batch_x1)
                
                loss = self.cfm.compute_loss(batch_x0, batch_x1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.training_losses.append(loss.item())
            if epoch % 5 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}, Loss: {loss.item()}")
    @torch.no_grad()
    def infer(self, num_samples=256, x0=None):
        """Run inference on the trained model.

        By default, inference starts from the same source sampler used during
        training to avoid train/infer distribution mismatch.
        """
        if x0 is None:
            x0 = self.source_sampler(
                num_samples,
                self.model_config.input_dim,
                self.flow_config.device,
            )
        else:
            x0 = x0.to(self.flow_config.device)
            num_samples = x0.shape[0]
        flow_output = infer.infer(self.model, x0, self.flow_config)
        return flow_output

    def infer_with_logprob(self, num_samples=256, x0=None, num_steps=None):
        """Generate samples and log p_theta(x) via CNF change of variables."""
        if x0 is None:
            x0 = self.source_sampler(
                num_samples,
                self.model_config.input_dim,
                self.flow_config.device,
            )
        else:
            x0 = x0.to(self.flow_config.device)
            num_samples = x0.shape[0]

        model_type = getattr(self.model_config, "model_type", "mlp")
        source_log_prob_fn = logprob.make_source_log_prob_fn(self.flow_config)
        x1, log_p1 = logprob.infer_with_logprob(
            self.model,
            x0,
            self.flow_config,
            source_log_prob_fn,
            model_type=model_type,
            num_steps=num_steps,
        )
        return x1, log_p1, x0
    
    def visualize_paths(self, flow_output=None):
        """Visualize integration paths."""
        if flow_output is None:
            flow_output = self.infer()
        
        flow_np = flow_output.detach().cpu().numpy()
        # expected shapes: [T, B, D] (time, batch, dim) or [T, B]
        T, B, D = flow_np.shape
        t = np.arange(T)
        plt.figure(figsize=(8, 5))
        for i in range(B):
            plt.plot(t, flow_np[:, i, 0], alpha=0.6)
        plt.xlabel("integration step")
        plt.ylabel("value")
        plt.title("Integration paths for each sample")
        plt.tight_layout()
        plt.show()
    
    def _flow_config_dict(self):
        return {
            "num_steps": self.flow_config.num_steps,
            "lr": self.flow_config.lr,
            "sigma_min": self.flow_config.sigma_min,
            "sigma_max": self.flow_config.sigma_max,
            "use_spectral": self.flow_config.use_spectral,
            "use_sobolev_ot": self.flow_config.use_sobolev_ot,
            "use_matern_source": self.flow_config.use_matern_source,
            "alpha": self.flow_config.alpha,
            "beta": self.flow_config.beta,
            "nu": self.flow_config.nu,
            "tau": self.flow_config.tau,
            "spectral_loss_modes": getattr(self.flow_config, "spectral_loss_modes", None),
            "spectral_loss_basis": getattr(self.flow_config, "spectral_loss_basis", "laplacian"),
            "grid_shape": getattr(self.flow_config, "grid_shape", None),
        }

    def _model_config_dict(self):
        return {
            "input_dim": self.model_config.input_dim,
            "output_dim": self.model_config.output_dim,
            "intermediate_dim": self.model_config.intermediate_dim,
            "enable_time": self.model_config.enable_time,
            "model_type": getattr(self.model_config, "model_type", "mlp"),
            "grid_shape": getattr(self.model_config, "grid_shape", None),
            "fno_modes": getattr(self.model_config, "fno_modes", 16),
            "fno_modes2": getattr(self.model_config, "fno_modes2", None),
            "fno_width": getattr(self.model_config, "fno_width", 64),
            "fno_layers": getattr(self.model_config, "fno_layers", 4),
            "fno_proj_dim": getattr(self.model_config, "fno_proj_dim", None),
            "fno_use_norm": getattr(self.model_config, "fno_use_norm", True),
            "spectral_modes": getattr(self.model_config, "spectral_modes", None),
            "sobolev_alpha": getattr(self.model_config, "sobolev_alpha", 0.0),
        }

    def save_checkpoint(self, path=None):
        """Save model weights and training metadata for later reload."""
        save_path = path or self.checkpoint_path
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        laplacian = self.flow_config.laplacian
        if laplacian is not None:
            laplacian = laplacian.detach().cpu()

        model_laplacian = getattr(self.model_config, "laplacian", None)
        if model_laplacian is not None:
            model_laplacian = model_laplacian.detach().cpu()

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self._model_config_dict(),
            "flow_config": self._flow_config_dict(),
            "laplacian": laplacian,
            "model_laplacian": model_laplacian,
            "training_losses": self.training_losses,
            "num_epochs": self.num_epochs,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        source_sampler=None,
        target_sampler=None,
        map_location=None,
    ):
        """Restore a trainer from a saved checkpoint."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location or "cpu",
            weights_only=False,
        )
        flow_config = config.FlowConfig(**checkpoint["flow_config"])
        laplacian = checkpoint.get("laplacian")
        if laplacian is not None:
            flow_config.laplacian = laplacian.to(flow_config.device)

        model_config_kwargs = dict(checkpoint["model_config"])
        model_laplacian = checkpoint.get("model_laplacian")
        if model_laplacian is not None:
            model_config_kwargs["laplacian"] = model_laplacian.to(flow_config.device)
        elif flow_config.laplacian is not None:
            model_config_kwargs["laplacian"] = flow_config.laplacian
        model_config = config.ModelConfig(**model_config_kwargs)
        trainer = cls(
            flow_config=flow_config,
            model_config=model_config,
            num_epochs=checkpoint.get("num_epochs", 100),
            dataset_size=checkpoint.get("dataset_size", 50000),
            batch_size=checkpoint.get("batch_size", 256),
            checkpoint_path=checkpoint_path,
            source_sampler=source_sampler,
            target_sampler=target_sampler,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.training_losses = checkpoint.get("training_losses", [])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return trainer


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    
    # test inference and visualization
    flow_output = trainer.infer(num_samples=256)
    trainer.visualize_paths(flow_output)
    
    # save checkpoint
    trainer.save_checkpoint()
