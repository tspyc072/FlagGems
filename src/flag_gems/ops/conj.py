import torch


def conj(input: torch.Tensor) -> torch.Tensor:
    """
    Returns a view of the input tensor with the conjugate flag set.
    For real tensors, returns the input itself.
    For complex tensors, returns a view sharing the same underlying storage.
    """
    if not input.is_complex():
        return input
    return input._conj()
