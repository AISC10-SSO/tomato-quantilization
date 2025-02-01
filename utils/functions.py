import torch
    
def safe_kl_div(base_probabilities: torch.Tensor, altered_probabilities: torch.Tensor) -> torch.Tensor:

    output = torch.zeros_like(base_probabilities)
    suitable_indices = (base_probabilities > 1e-3) & (altered_probabilities > 1e-3)
    output[suitable_indices] = altered_probabilities[suitable_indices] * torch.log(altered_probabilities[suitable_indices] / base_probabilities[suitable_indices])

    return output.sum(dim=-1)

def safe_exp_logits(x: torch.Tensor) -> torch.Tensor:

    x = x - torch.max(x, dim=-2, keepdim=True).values.detach()

    return x.exp()

def normalize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return probabilities / probabilities.sum(dim=-1, keepdim=True)