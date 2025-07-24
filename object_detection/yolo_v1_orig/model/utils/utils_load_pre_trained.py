import numpy as np
import os
import torch
import torch.nn as nn

from model.yolov1 import YOLOv1

# TODO find a yolov1 pre-trained model

def load_pretrained_weights(model: nn.Module, weight_filename: str):
    """
    Loads pre-trained Darknet weights into a PyTorch model.
    NOTE: Assumes the model architecture matches the weight file.
    """
    cwd = os.getcwd()
    file_path = os.path.join(cwd, f"model/pre_trained/{weight_filename}")
    # Open and read the binary weight file
    with open(file_path, "rb") as f:
        # The first 5 values are header information:
        # 1. Major version
        # 2. Minor version
        # 3. Subversion
        # 4, 5. Images seen by the network during training
        header = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)  # The rest are the weights

    # Pointer to track our position in the weights array
    ptr = 0

    # Get all modules in my model's architecture; the conv layers needs to be the same as the pre-trained model.
    all_modules = [module for module in model.modules()]

    for i, module in enumerate(all_modules):
        # We only load weights for Convolutional layers
        if isinstance(module, nn.Conv2d):
            # Check if the convolutional layer has batch normalization
            try:
                # Look ahead to see if the next module is BatchNorm
                next_module = all_modules[i + 1]
                has_batch_norm = isinstance(next_module, nn.BatchNorm2d)
            except IndexError:
                has_batch_norm = False

            if has_batch_norm:
                bn_module = next_module  # Ex: BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                # 1. Load BatchNorm biases
                num_b = bn_module.bias.numel()
                # Load the pre_trained BatchNorm biases.
                bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                    bn_module.bias
                )
                bn_module.bias.data.copy_(bn_b)
                ptr += num_b

                # 2. Load BatchNorm weights
                num_w = bn_module.weight.numel()
                bn_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(
                    bn_module.weight
                )
                bn_module.weight.data.copy_(bn_w)
                ptr += num_w

                # 3. Load BatchNorm running mean
                num_rm = bn_module.running_mean.numel()
                bn_rm = torch.from_numpy(weights[ptr : ptr + num_rm]).view_as(
                    bn_module.running_mean
                )
                bn_module.running_mean.data.copy_(bn_rm)
                ptr += num_rm

                # 4. Load BatchNorm running variance
                num_rv = bn_module.running_var.numel()
                bn_rv = torch.from_numpy(weights[ptr : ptr + num_rv]).view_as(
                    bn_module.running_var
                )
                bn_module.running_var.data.copy_(bn_rv)
                ptr += num_rv
            else:
                # If no batch norm, load the convolutional bias
                num_b = module.bias.numel()
                conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(
                    module.bias
                )
                module.bias.data.copy_(conv_b)
                ptr += num_b
            # Finally, load the convolutional weights
            num_w = module.weight.numel()
            conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(module.weight)
            module.weight.data.copy_(conv_w)
            ptr += num_w


# Test as a module:
#   python -m model.utils.utils_pre_trained
def test():
    from configs.config_loader import load_config

    cfg = load_config("config_voc_dataset.yaml")
    model = YOLOv1(cfg=cfg, in_channels=3)
    load_pretrained_weights(model, "darknet19_448.conv.23")
    print("Pre-trained convolutional weights loaded.")


if __name__ == "__main__":
    test()
