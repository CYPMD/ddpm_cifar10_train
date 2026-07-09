import os
import copy
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion_JIT_Train.DiffusionJIT import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion_JIT_Train.ModelJIT import UNet

# ---------------------------------------------------------
# A100 OPTIMIZATIONS: Enable TF32 for Tensor Cores
# ---------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    # def update_model_average(self, ma_model, current_model):
    #     for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
    #         old_weight, up_weight = ma_params.data, current_params.data
    #         ma_params.data = self.update_average(old_weight, up_weight)
    @torch.no_grad()
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            # In-place multiplication and addition to completely prevent VRAM fragmentation
            ma_params.data.mul_(self.beta).add_(current_params.data, alpha=1.0 - self.beta)

    # def update_average(self, old, new):
    #     if old is None:
    #         return new
    #     return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=modelConfig["batch_size"], 
        shuffle=True, 
        num_workers=8, 
        persistent_workers=True, 
        drop_last=True, 
        pin_memory=True
    )

    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    
    # Deepcopy the original model FIRST, before it gets wrapped by torch.compile!
    ema_model = copy.deepcopy(net_model).eval().requires_grad_(False)
    
    print("Compiling model for A100... (This takes a minute)")
    net_model = torch.compile(net_model, mode="max-autotune")
    ema = EMA(modelConfig["ema_decay"])
    
    optimizer = torch.optim.Adam(net_model.parameters(), lr=modelConfig["lr"])
    
    # Passing the decoupled prediction space and loss space
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
        pred_space=modelConfig.get("pred_space", "eps"),
        loss_space=modelConfig.get("loss_space", "eps")
    ).to(device)

    scaler = torch.cuda.amp.GradScaler()

    step = 0
    with tqdm(initial=step, total=modelConfig["training_steps"], dynamic_ncols=True) as pbar:
        while step < modelConfig["training_steps"]:
            for images, labels in dataloader:
                optimizer.zero_grad(set_to_none=True)
                x_0 = images.to(device)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = trainer(x_0).mean() 
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
                
                scaler.step(optimizer)
                scaler.update()
                
                # Update EMA using the underlying model (bypass torch.compile wrapper)
                underlying_net = net_model._orig_mod if hasattr(net_model, "_orig_mod") else net_model
                underlying_ema = ema_model._orig_mod if hasattr(ema_model, "_orig_mod") else ema_model
                ema.step_ema(underlying_ema, underlying_net)
                
                step += 1
                pbar.update(1)
                pbar.set_postfix(ordered_dict={
                    "loss": f"{loss.item():.4f}",
                    "step": step
                })
                
                if step % modelConfig["save_weight_interval"] == 0 or step == modelConfig["training_steps"]:
                    torch.save(underlying_net.state_dict(), os.path.join(
                        modelConfig["save_weight_dir"], f'ckpt_{step}.pt'))
                    torch.save(underlying_ema.state_dict(), os.path.join(
                        modelConfig["save_weight_dir"], f'ckpt_ema_{step}.pt'))
                
                if step >= modelConfig["training_steps"]:
                    break


def eval(modelConfig: Dict):
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.).to(device)
        
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("Model weight load done.")
        
        model = torch.compile(model)
        model.eval()
        
        # Sampler only needs to know what the network outputs (`pred_space`)
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
            pred_objective=modelConfig.get("pred_space", "eps")
        ).to(device)
        
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        
        with torch.cuda.amp.autocast():
            sampledImgs = sampler(noisyImage)
            
        sampledImgs = sampledImgs * 0.5 + 0.5  
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
        print("Images generated and saved!")
