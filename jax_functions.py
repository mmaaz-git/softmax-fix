"""JAX baselines, Barron et al., and our log1p fix (last axis, no reduction).

The barron_* functions reproduce Sections 2 and 3 of
https://arxiv.org/html/2610.31642v1, changing only their names/references.
Authors: Jonathan T. Barron, Dan Gnanapragasam, and Michael Tschannen.
Source license: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Differentiate soft-target losses with respect to the student (argument 0).
The cancellation fixes supply the paper's first-order student gradient.
Precision is controlled by the caller; test_jax.py enables float64 explicitly.
"""

import jax
import jax.numpy as jnp


# Standard JAX. Cross-entropy is composed from jax.nn.log_softmax;
# the soft-target expression is the baseline used in Section 3 of the paper.

def standard_log_softmax(x):
    return jax.nn.log_softmax(x, axis=-1)

def standard_softmax(x):
    return jax.nn.softmax(x, axis=-1)

def standard_cross_entropy(z, t):
    return -jnp.take_along_axis(standard_log_softmax(z), t[..., None], axis=-1).squeeze(-1)

def standard_soft_target_cross_entropy(student, teacher):
    p = standard_softmax(jax.lax.stop_gradient(teacher))
    return -jnp.sum(p * standard_log_softmax(student), axis=-1)


# Barron et al., arXiv:2610.31642v1 (not the mutable project website).

def barron_log_softmax(z):
    i_max = jnp.argmax(z, axis=-1)
    z_max = jnp.take_along_axis(z, i_max[..., None], axis=-1)
    is_max = jax.nn.one_hot(i_max, z.shape[-1], dtype=bool)
    shifted = jnp.where(is_max, 0.0, z - z_max)
    others = jnp.where(is_max, -jnp.inf, shifted)
    m = -jax.nn.logsumexp(others, axis=-1, keepdims=True)
    return shifted - jax.nn.softplus(-m)

barron_softmax = lambda z: jnp.exp(barron_log_softmax(z))

def barron_cross_entropy(z, t):
    z_t = jnp.take_along_axis(z, t[..., None], axis=-1)
    is_t = jax.nn.one_hot(t, z.shape[-1], dtype=bool)
    others = jnp.where(is_t, -jnp.inf, z - z_t)
    m = -jax.nn.logsumexp(others, axis=-1)
    return jax.nn.softplus(-m)

def barron_soft_target_cross_entropy(z_S, z_T):
    log_q = barron_log_softmax(z_S)
    q, p = jnp.exp(log_q), barron_softmax(z_T)
    delta = z_S - z_T
    d = delta - jnp.sum(p * delta, axis=-1, keepdims=True)
    s = d - jnp.log1p(jnp.sum(p * jnp.expm1(d), axis=-1, keepdims=True))
    u = p * jnp.expm1(s)
    g = u - q * jnp.sum(u, axis=-1, keepdims=True)
    sg = jax.lax.stop_gradient
    loss = -jnp.sum(p * sg(log_q), axis=-1)
    return loss + jnp.sum(sg(g) * (z_S - sg(z_S)), axis=-1)


# Our replacement: sum the non-maximum exponentials, then use log1p.

def new_log_softmax(x):
    i_max = jnp.argmax(x, axis=-1)
    x_max = jnp.take_along_axis(x, i_max[..., None], axis=-1)
    is_max = jax.nn.one_hot(i_max, x.shape[-1], dtype=bool)
    shifted = jnp.where(is_max, 0.0, x - x_max)
    others = jnp.where(is_max, -jnp.inf, shifted)
    normalizer = jnp.log1p(jnp.exp(others).sum(axis=-1, keepdims=True))
    return shifted - normalizer

def new_softmax(x):
    return jnp.exp(new_log_softmax(x))

def new_cross_entropy(z, t):
    return -jnp.take_along_axis(new_log_softmax(z), t[..., None], axis=-1).squeeze(-1)

def new_soft_target_cross_entropy(student, teacher):
    teacher = jax.lax.stop_gradient(teacher)
    log_q = new_log_softmax(student)
    q = jnp.exp(log_q)
    p = new_softmax(teacher)
    delta = student - teacher
    d = delta - jnp.sum(p * delta, axis=-1, keepdims=True)
    s = d - jnp.log1p(jnp.sum(p * jnp.expm1(d), axis=-1, keepdims=True))
    u = p * jnp.expm1(s)
    g = u - q * jnp.sum(u, axis=-1, keepdims=True)
    sg = jax.lax.stop_gradient
    loss = -jnp.sum(p * sg(log_q), axis=-1)
    return loss + jnp.sum(sg(g) * (student - sg(student)), axis=-1)
