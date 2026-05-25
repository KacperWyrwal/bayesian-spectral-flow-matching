import os, sys; 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import matplotlib.pyplot as plt
import numpy as np
import Flows.FM as FM
import Flows.config as config
import models.architecture as MA
import Flows.infer as infer
import Dataset.MoG as MoG


class Trainer:
    """Trainer class for Flow Matching models."""
    
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
        self.model = MA.dummy_mlp(self.model_config).to(self.flow_config.device)
        self.cfm = FM.ConditionalFM(self.model, self.flow_config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.flow_config.lr
        )
        self.dataloader = None
        self.training_losses = []
    
    def _default_source_sampler(self, num_samples, input_dim, device):
        """Default source distribution: standard Gaussian."""
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
                
                loss = self.cfm.compute_loss(batch_x0, batch_x1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.training_losses.append(loss.item())
            if epoch % 5 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}, Loss: {loss.item()}")
    
    def infer(self, num_samples=256):
        """Run inference on the trained model."""
        x = torch.randn(num_samples, self.model_config.input_dim).to(
            self.flow_config.device
        )
        flow_output = infer.infer(self.model, x, self.flow_config)
        return flow_output
    
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
    
    def save_checkpoint(self, path=None):
        """Save model checkpoint."""
        save_path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    
    # test inference and visualization
    flow_output = trainer.infer(num_samples=256)
    trainer.visualize_paths(flow_output)
    
    # save checkpoint
    trainer.save_checkpoint()
