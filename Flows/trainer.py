import os, sys; 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import matplotlib.pyplot as plt
import numpy as np
import FM as FM
import config as config
import models.architecture as MA
import Flows.infer as infer
import Dataset.MoG as MoG



if __name__ == "__main__":
    Flowconfig = config.FlowConfig(num_steps=10, lr=5e-4,sigma_min=0.01, sigma_max=1.0)
    model_config = config.ModelConfig(input_dim=1, output_dim=1, intermediate_dim=256)
    print(f"Device: {Flowconfig.device}")
    
    model = MA.dummy_mlp(model_config).to(model_config.device)
    cfm = FM.ConditionalFM(model,Flowconfig)
    # --- Batched gradient descent using DataLoader ---
    dataset_size = 16384
    batch_size = 64
    # sample x0 from a standard Gaussian and x1 from the mixture model
    x0_all = torch.randn(dataset_size, model_config.input_dim, device=Flowconfig.device)
    x1_all = MoG.sample_mog(dataset_size).unsqueeze(1).to(Flowconfig.device)
    dataset = torch.utils.data.TensorDataset(x0_all, x1_all)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=Flowconfig.lr)
    num_epochs = 1000
    for epoch in range(num_epochs):
        for batch_idx, (batch_x0, batch_x1) in enumerate(dataloader):
            batch_x0 = batch_x0.to(Flowconfig.device)
            batch_x1 = batch_x1.to(Flowconfig.device)

            loss = cfm.compute_loss(batch_x0, batch_x1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} Batch {batch_idx}, Loss: {loss.item()}")
    # test inference
    x = torch.randn(64, 1).to(Flowconfig.device)
    flow_output = infer.infer(model, x, Flowconfig)

    # visualize the integration paths across time for each sample
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
    # save checkpoint
    torch.save(model.state_dict(), "./checkpoints/cfm_checkpoint.pth")
