import jax
import jax.numpy as jnp
import numpy as np

from optiq.transport import (
    gradient_skewed_mixture_log_density,
    sample_gradient_skewed_mixture,
)


def test_skewed_samples_respect_radius_and_follow_positive_gradient():
    centers = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    gradients = jnp.asarray([[[1.26, 0.0]]], dtype=jnp.float32)
    samples = sample_gradient_skewed_mixture(
        jax.random.PRNGKey(0),
        centers,
        gradients,
        repeats=20_000,
        std=0.2,
        perturb_clip=0.5,
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )
    radii = jnp.linalg.norm(samples, axis=-1)
    forward_fraction = jnp.mean(samples[..., 0] > 0.0)
    assert float(jnp.max(radii)) <= 0.50001
    assert 0.74 < float(forward_fraction) < 0.80


def test_skewed_density_is_numerically_normalized_in_two_dimensions():
    centers = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    gradients = jnp.asarray([[[1.26, 0.0]]], dtype=jnp.float32)
    coordinates = jnp.linspace(-0.51, 0.51, 501)
    xx, yy = jnp.meshgrid(coordinates, coordinates)
    samples = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)[None]
    log_density = gradient_skewed_mixture_log_density(
        samples,
        centers,
        gradients,
        std=0.2,
        perturb_clip=0.5,
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )
    density = np.asarray(jnp.exp(log_density)).reshape(xx.shape)
    mass = np.trapezoid(
        np.trapezoid(density, x=np.asarray(coordinates), axis=1),
        x=np.asarray(coordinates),
    )
    assert np.isclose(mass, 1.0, atol=2.0e-3)


def test_zero_gradient_recovers_a_symmetric_proposal():
    centers = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    gradients = jnp.zeros_like(centers)
    samples = sample_gradient_skewed_mixture(
        jax.random.PRNGKey(2),
        centers,
        gradients,
        repeats=20_000,
        std=0.2,
        perturb_clip=0.5,
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )
    positive_fraction = float(jnp.mean(samples[..., 0] > 0.0))
    assert 0.48 < positive_fraction < 0.52
