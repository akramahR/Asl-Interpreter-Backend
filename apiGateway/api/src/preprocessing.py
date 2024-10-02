import os
import json
import torch
import numpy as np
import cv2


def processFramesToTensor(frames):
    tensor = torch.from_numpy(frames)
    tensor = tensor_normalize(tensor)
    tensor = torch.permute(tensor, (3, 0, 1, 2))
    tensor = torch.reshape(tensor, (1, 1, tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3]))
    return tensor




def tensor_normalize(tensor):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
