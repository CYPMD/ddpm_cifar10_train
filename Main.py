import os
from Diffusion.Train import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "eval", # or "eval" "train"
        "training_steps": 100000,     # CIFAR-10 paper trained for 800k steps exactly
        "save_weight_interval": 10000,# Save weights every 50k steps to save disk space
        "batch_size": 1024,           # Match paper
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2], # Match paper
        "attn": [1],                  # Attention at 16x16 resolution
        "num_res_blocks": 2,
        "dropout": 0.1,               # Match paper for CIFAR-10
        "lr": 5e-4,                   # Constant learning rate from the paper
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "ema_decay": 0.9999,          # EMA decay rate from paper
        "device": "cuda:0", 
        "training_load_weight": None, 
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_ema_40000.pt", # Evaluate on the final EMA weights
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs_ema_40000.png",
        "sampledImgName": "SampledNoGuidenceImgs_ema_40000.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    
    # Create directories if they don't exist
    os.makedirs(modelConfig["save_weight_dir"], exist_ok=True)
    os.makedirs(modelConfig["sampled_dir"], exist_ok=True)

    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)

if __name__ == '__main__':
    main()
