import sys

import jax
import jax.numpy as jnp
import pytest
from hypothesis import example, given, strategies as st, settings

from jax_functions import standard_softmax, barron_softmax, new_softmax
from jax_functions import standard_soft_target_cross_entropy, barron_soft_target_cross_entropy, new_soft_target_cross_entropy

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
@given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
@example(xs=[0.0, 37.0])
@example(xs=[1.7976931348623157e308, -9.9792015476736e291])
@settings(max_examples=10_000, deadline=None)
def test_softmax_gradients_sum_to_zero(softmax, xs):
    x = jnp.array(xs, dtype=jnp.float64)
    J = jax.jacrev(softmax)(x)
    assert (jnp.abs(J.sum(axis=1)) <= 1e-12 * jnp.abs(J).sum(axis=1)).all()


@pytest.mark.parametrize("soft_target_cross_entropy", [standard_soft_target_cross_entropy, barron_soft_target_cross_entropy, new_soft_target_cross_entropy], ids=["standard", "barron", "new"])
@given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
@example(xs=[0.0, 1.0])
@settings(max_examples=10_000, deadline=None)
def test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits(soft_target_cross_entropy, xs):
    x = jnp.array(xs, dtype=jnp.float64)
    grad = jax.grad(soft_target_cross_entropy, argnums=0)(x, jax.lax.stop_gradient(x))
    assert jnp.all(grad == 0)
