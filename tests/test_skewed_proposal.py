import jax
import jax.numpy as jnp
import numpy as np

from optiq.transport import (
    gradient_perpendicular_mixture_log_density,
    gradient_skewed_mixture_log_density,
    sample_gradient_perpendicular_mixture,
    sample_gradient_skewed_mixture,
    sample_truncated_gaussian_mixture,
    truncated_mixture_log_density,
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


def test_exact_truncated_gaussian_mixture_matches_its_density():
    centers = jnp.asarray([[[-0.5], [0.5]]], dtype=jnp.float32)
    samples = sample_truncated_gaussian_mixture(
        jax.random.PRNGKey(3),
        centers,
        repeats=10_000,
        std=0.1,
        perturb_clip=0.25,
        action_low=(-1.0,),
        action_high=(1.0,),
    ).reshape(1, -1, 1)
    assert 0.48 < float(jnp.mean(samples > 0.0)) < 0.52
    assert float(jnp.min(samples)) >= -0.75001
    assert float(jnp.max(samples)) <= 0.75001

    coordinates = jnp.linspace(-1.0, 1.0, 4001)[None, :, None]
    density = jnp.exp(
        truncated_mixture_log_density(
            coordinates,
            centers,
            std=0.1,
            perturb_clip=0.25,
            action_low=(-1.0,),
            action_high=(1.0,),
        )
    )[0]
    mass = np.trapezoid(np.asarray(density), x=np.asarray(coordinates[0, :, 0]))
    assert np.isclose(mass, 1.0, atol=2.0e-3)


def test_perpendicular_proposal_is_full_rank_and_keeps_backward_mass():
    centers = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    gradients = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32)
    samples = sample_gradient_perpendicular_mixture(
        jax.random.PRNGKey(4),
        centers,
        gradients,
        repeats=20_000,
        std=0.2,
        perturb_clip=0.5,
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
    )[0, 0]
    standardized_radius = jnp.sqrt(
        jnp.square((samples[:, 0] - 0.05) / 0.05)
        + jnp.square(samples[:, 1] / 0.2)
    )
    backward_fraction = float(jnp.mean(samples[:, 0] < 0.0))
    assert float(jnp.max(standardized_radius)) <= 2.50001
    assert 0.05 < backward_fraction < 0.25
    assert float(jnp.var(samples[:, 1])) > 8.0 * float(jnp.var(samples[:, 0]))


def test_perpendicular_density_is_numerically_normalized():
    centers = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    gradients = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32)
    x_coordinates = jnp.linspace(-0.09, 0.19, 601)
    y_coordinates = jnp.linspace(-0.52, 0.52, 601)
    xx, yy = jnp.meshgrid(x_coordinates, y_coordinates)
    samples = jnp.stack((xx.ravel(), yy.ravel()), axis=-1)[None]
    density = np.asarray(
        jnp.exp(
            gradient_perpendicular_mixture_log_density(
                samples,
                centers,
                gradients,
                std=0.2,
                perturb_clip=0.5,
                action_low=(-1.0, -1.0),
                action_high=(1.0, 1.0),
            )
        )
    ).reshape(xx.shape)
    mass = np.trapezoid(
        np.trapezoid(density, x=np.asarray(x_coordinates), axis=1),
        x=np.asarray(y_coordinates),
    )
    assert np.isclose(mass, 1.0, atol=3.0e-3)
