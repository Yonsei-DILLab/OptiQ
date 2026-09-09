"""Shared, deterministic evaluation for GMM-40 sampler comparisons."""

import torch

def nearest_mode(targets, mode_locations):
    return torch.cdist(targets, mode_locations).argmin(dim=1)


def empirical_1d_distances(samples: torch.Tensor, reference: torch.Tensor):
    """Return equal-sample empirical W1 and two-sample KS distances."""
    if samples.ndim != 1 or reference.ndim != 1:
        raise ValueError("empirical samples must be one-dimensional")
    if samples.shape != reference.shape or samples.numel() < 2:
        raise ValueError("empirical samples must have the same non-trivial shape")
    sorted_samples = samples.sort().values
    sorted_reference = reference.sort().values
    wasserstein_1 = (sorted_samples - sorted_reference).abs().mean()

    support = torch.cat((sorted_samples, sorted_reference)).sort().values
    count = float(samples.numel())
    sample_cdf_left = torch.searchsorted(
        sorted_samples, support, right=False
    ).to(samples.dtype) / count
    sample_cdf_right = torch.searchsorted(
        sorted_samples, support, right=True
    ).to(samples.dtype) / count
    reference_cdf_left = torch.searchsorted(
        sorted_reference, support, right=False
    ).to(reference.dtype) / count
    reference_cdf_right = torch.searchsorted(
        sorted_reference, support, right=True
    ).to(reference.dtype) / count
    kolmogorov_smirnov = torch.maximum(
        (sample_cdf_left - reference_cdf_left).abs().max(),
        (sample_cdf_right - reference_cdf_right).abs().max(),
    )
    return wasserstein_1, kolmogorov_smirnov


@torch.no_grad()
def evaluate_sample_tensor(
    samples: torch.Tensor,
    target,
    reference_seed: int = 20260821,
):
    """Evaluate fixed samples against one deterministic reference draw."""
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("GMM-40 samples must have shape [N, 2]")
    if samples.shape[0] < 2:
        raise ValueError("at least two samples are required")
    device = samples.device
    cuda_devices = [torch.cuda.current_device()] if samples.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(reference_seed)
        if samples.is_cuda:
            torch.cuda.manual_seed_all(reference_seed)
        reference = target.sample((samples.shape[0],)).to(device)

    assignments = nearest_mode(samples, target.locs)
    counts = torch.bincount(assignments, minlength=target.n_mixes).float()
    occupancy = counts / counts.sum()

    projection_generator = torch.Generator(device=device)
    projection_generator.manual_seed(reference_seed + 1)
    directions = torch.randn(
        128, 2, generator=projection_generator, device=device
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    generated_projection = (samples @ directions.T).sort(dim=0).values
    reference_projection = (reference @ directions.T).sort(dim=0).values
    sliced_w2 = (generated_projection - reference_projection).square().mean().sqrt()

    sample_log_prob = target.log_prob(samples)
    reference_log_prob = target.log_prob(reference)
    energy_w1, energy_ks = empirical_1d_distances(
        -sample_log_prob, -reference_log_prob
    )
    low_density_cutoff = torch.quantile(reference_log_prob, 0.01)
    nearest_distance = torch.cdist(samples, target.locs).min(dim=1).values
    return {
        "eval/sample_count": samples.shape[0],
        "eval/mean_log_prob": sample_log_prob.mean().item(),
        "eval/log_prob_std": sample_log_prob.std().item(),
        "eval/mean_log_prob_abs_error": (
            sample_log_prob.mean() - reference_log_prob.mean()
        ).abs().item(),
        "eval/energy_w1": energy_w1.item(),
        "eval/energy_ks": energy_ks.item(),
        "eval/modes_covered": int((counts > 0).sum().item()),
        "eval/mode_occupancy_tvd": (
            0.5 * (occupancy - 1.0 / target.n_mixes).abs().sum()
        ).item(),
        "eval/sliced_w2": sliced_w2.item(),
        "eval/mean_nearest_mode_distance": nearest_distance.mean().item(),
        "eval/low_density_fraction": (
            sample_log_prob < low_density_cutoff
        ).float().mean().item(),
        "eval/reference_mean_log_prob": reference_log_prob.mean().item(),
        "eval/reference_log_prob_std": reference_log_prob.std().item(),
        "eval/reference_low_density_cutoff": low_density_cutoff.item(),
    }
