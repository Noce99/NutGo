import torch


def my_argmax(tensor2d):
    max_index = torch.argmax(tensor2d)
    row = max_index // tensor2d.shape[0]
    col = max_index - row*tensor2d.shape[0]
    return int(row), int(col)
