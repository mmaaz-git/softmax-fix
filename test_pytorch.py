import sys

import pytest
import torch
from hypothesis import example, given, strategies as st, settings

from pytorch_functions import standard_softmax, barron_softmax, new_softmax
from pytorch_functions import standard_soft_target_cross_entropy, barron_soft_target_cross_entropy, new_soft_target_cross_entropy


@pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
@given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
@example(xs=[0.0, 37.0])
@example(xs=[1.7976931348623157e308, -9.9792015476736e291])
@settings(max_examples=10_000, deadline=None)
def test_softmax_gradients_sum_to_zero(softmax, xs):
    x = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
    J = torch.autograd.functional.jacobian(softmax, x)
    assert (J.sum(dim=1).abs() <= 1e-12 * J.abs().sum(dim=1)).all()


@pytest.mark.parametrize("soft_target_cross_entropy", [standard_soft_target_cross_entropy, barron_soft_target_cross_entropy, new_soft_target_cross_entropy], ids=["standard", "barron", "new"])
@given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
@example(xs=[0.0, 1.0])
@settings(max_examples=10_000, deadline=None)
def test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits(soft_target_cross_entropy, xs):
    x = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
    loss = soft_target_cross_entropy(x, x.detach())
    grad, = torch.autograd.grad(loss, x)
    assert (grad == 0).all()
