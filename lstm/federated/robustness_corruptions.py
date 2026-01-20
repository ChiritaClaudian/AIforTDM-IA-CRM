import torch

def gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0:
        return x
    return x + sigma * torch.randn_like(x)

def channel_dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    # x: (B, T, C=3)
    if p <= 0:
        return x
    B, T, C = x.shape
    mask = (torch.rand(B, C, device=x.device) > p).float()  # (B,C)
    return x * mask[:, None, :]

def time_mask(x: torch.Tensor, length: int) -> torch.Tensor:
    if length <= 0:
        return x
    B, T, C = x.shape
    length = min(length, T)
    start = torch.randint(0, T - length + 1, (B,), device=x.device)
    x2 = x.clone()
    for i in range(B):
        x2[i, start[i]:start[i] + length, :] = 0
    return x2

def time_shift(x: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return x
    # circular shift; alternatively do zero-pad shift
    return torch.roll(x, shifts=k, dims=1)
