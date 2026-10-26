# softmax-fix

Property-based tests for two properties of softmax and cross-entropy.

The two properties tested are:

1. Each row of the softmax Jacobian sums to zero, with a relative tolerance of `1e-12`.
2. When student and teacher logits match, the cross-entropy gradient with respect to the student logits is zero, holding the teacher fixed.

You can test the standard implementations in PyTorch and JAX, or the implementations from [Barron et al.](https://arxiv.org/html/2610.31642v1), or our modified implementation.

The PyTorch implementations are in `pytorch_functions.py`, with three different versions of softmax, log-softmax, and cross-entropy (with hard or soft targets). In particular, it exposes the functions:

- `standard_softmax`, `barron_softmax`, and `new_softmax`.
- `standard_log_softmax`, `barron_log_softmax`, and `new_log_softmax`.
- `standard_cross_entropy`, `barron_cross_entropy`, and `new_cross_entropy` for hard targets.
- `standard_soft_target_cross_entropy`, `barron_soft_target_cross_entropy`, and `new_soft_target_cross_entropy` for soft targets.

The JAX implementations are in `jax_functions.py` and expose the same function names.

## Run the tests

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked
uv run --locked pytest -v
```

You can run the test files (`test_pytorch.py` or `test_jax.py`) simply with `pytest`.

```sh
uv run --locked pytest test_pytorch.py
```

Use `-k` to select `standard`, `barron`, or `new` (our implementation); for example, to run the standard implementations for JAX:

```sh
uv run --locked pytest test_jax.py -k standard
```

Known counterexamples are already saved in the test files as `@example(...)` decorators for reproducibility and are replayed by default, so some tests are expected to fail. If you would like to start a fresh search, simply add `--search`:

```sh
uv run --locked pytest --search -v
```

Fresh searches skip the saved examples and local cache, trying up to 10,000 examples per property. They can take several minutes.


## Log output

Full output from replaying the saved examples in both test files:

```text
$ uv run --locked pytest test_pytorch.py test_jax.py
============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-8.4.1, pluggy-1.7.0
rootdir: /Users/maaz/Documents/Research/softmax-fix
configfile: pyproject.toml
plugins: hypothesis-6.136.7
collected 12 items

test_pytorch.py FF.F..                                                   [ 50%]
test_jax.py FF.F..                                                       [100%]

=================================== FAILURES ===================================
_________________ test_softmax_gradients_sum_to_zero[standard] _________________

softmax = <function standard_softmax at 0x10e188fe0>

    @pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
>   @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
                   ^^^^^^^^^^

test_pytorch.py:12: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
.venv/lib/python3.13/site-packages/hypothesis/core.py:1596: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

softmax = <function standard_softmax at 0x10e188fe0>, xs = [0.0, 37.0]

    @pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
    @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
    @example(xs=[0.0, 37.0])
    @example(xs=[1.7976931348623157e308, -9.9792015476736e291])
    @settings(max_examples=10_000, deadline=None)
    def test_softmax_gradients_sum_to_zero(softmax, xs):
        x = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
        J = torch.autograd.functional.jacobian(softmax, x)
>       assert (J.sum(dim=1).abs() <= 1e-12 * J.abs().sum(dim=1)).all()
E       assert tensor(False)
E        +  where tensor(False) = <built-in method all of Tensor object at 0x111e29770>()
E        +    where <built-in method all of Tensor object at 0x111e29770> = tensor([1.2326e-32, 8.5330e-17], dtype=torch.float64) <= (1e-12 * tensor([1.7066e-16, 8.5330e-17], dtype=torch.float64)).all
E        +      where tensor([1.2326e-32, 8.5330e-17], dtype=torch.float64) = <built-in method abs of Tensor object at 0x111e293b0>()
E        +        where <built-in method abs of Tensor object at 0x111e293b0> = tensor([-1.2326e-32, -8.5330e-17], dtype=torch.float64).abs
E        +          where tensor([-1.2326e-32, -8.5330e-17], dtype=torch.float64) = <built-in method sum of Tensor object at 0x111e294f0>(dim=1)
E        +            where <built-in method sum of Tensor object at 0x111e294f0> = tensor([[ 8.5330e-17, -8.5330e-17],\n        [-8.5330e-17,  0.0000e+00]], dtype=torch.float64).sum
E        +      and   tensor([1.7066e-16, 8.5330e-17], dtype=torch.float64) = <built-in method sum of Tensor object at 0x111e29540>(dim=1)
E        +        where <built-in method sum of Tensor object at 0x111e29540> = tensor([[8.5330e-17, 8.5330e-17],\n        [8.5330e-17, 0.0000e+00]], dtype=torch.float64).sum
E        +          where tensor([[8.5330e-17, 8.5330e-17],\n        [8.5330e-17, 0.0000e+00]], dtype=torch.float64) = <built-in method abs of Tensor object at 0x111e294f0>()
E        +            where <built-in method abs of Tensor object at 0x111e294f0> = tensor([[ 8.5330e-17, -8.5330e-17],\n        [-8.5330e-17,  0.0000e+00]], dtype=torch.float64).abs
E       Falsifying explicit example: test_softmax_gradients_sum_to_zero(
E           softmax=standard_softmax,
E           xs=[0.0, 37.0],
E       )

test_pytorch.py:19: AssertionError
__________________ test_softmax_gradients_sum_to_zero[barron] __________________

softmax = <function barron_softmax at 0x10e1893a0>

    @pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
>   @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
                   ^^^^^^^^^^

test_pytorch.py:12: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
.venv/lib/python3.13/site-packages/hypothesis/core.py:1596: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

softmax = <function barron_softmax at 0x10e1893a0>
xs = [1.7976931348623157e+308, -9.9792015476736e+291]

    @pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
    @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
    @example(xs=[0.0, 37.0])
    @example(xs=[1.7976931348623157e308, -9.9792015476736e291])
    @settings(max_examples=10_000, deadline=None)
    def test_softmax_gradients_sum_to_zero(softmax, xs):
        x = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
        J = torch.autograd.functional.jacobian(softmax, x)
>       assert (J.sum(dim=1).abs() <= 1e-12 * J.abs().sum(dim=1)).all()
E       assert tensor(False)
E        +  where tensor(False) = <built-in method all of Tensor object at 0x111e7b5c0>()
E        +    where <built-in method all of Tensor object at 0x111e7b5c0> = tensor([nan, nan], dtype=torch.float64) <= (1e-12 * tensor([nan, nan], dtype=torch.float64)).all
E        +      where tensor([nan, nan], dtype=torch.float64) = <built-in method abs of Tensor object at 0x111e7b840>()
E        +        where <built-in method abs of Tensor object at 0x111e7b840> = tensor([nan, nan], dtype=torch.float64).abs
E        +          where tensor([nan, nan], dtype=torch.float64) = <built-in method sum of Tensor object at 0x111e7b980>(dim=1)
E        +            where <built-in method sum of Tensor object at 0x111e7b980> = tensor([[nan, nan],\n        [nan, nan]], dtype=torch.float64).sum
E        +      and   tensor([nan, nan], dtype=torch.float64) = <built-in method sum of Tensor object at 0x111e7b700>(dim=1)
E        +        where <built-in method sum of Tensor object at 0x111e7b700> = tensor([[nan, nan],\n        [nan, nan]], dtype=torch.float64).sum
E        +          where tensor([[nan, nan],\n        [nan, nan]], dtype=torch.float64) = <built-in method abs of Tensor object at 0x111e7b980>()
E        +            where <built-in method abs of Tensor object at 0x111e7b980> = tensor([[nan, nan],\n        [nan, nan]], dtype=torch.float64).abs
E       Falsifying explicit example: test_softmax_gradients_sum_to_zero(
E           softmax=barron_softmax,
E           xs=[1.7976931348623157e+308, -9.9792015476736e+291],
E       )

test_pytorch.py:19: AssertionError
_ test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits[standard] _

soft_target_cross_entropy = <function standard_soft_target_cross_entropy at 0x10e189300>

    @pytest.mark.parametrize("soft_target_cross_entropy", [standard_soft_target_cross_entropy, barron_soft_target_cross_entropy, new_soft_target_cross_entropy], ids=["standard", "barron", "new"])
>   @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

test_pytorch.py:23: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
.venv/lib/python3.13/site-packages/hypothesis/core.py:1596: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

soft_target_cross_entropy = <function standard_soft_target_cross_entropy at 0x10e189300>
xs = [0.0, 1.0]

    @pytest.mark.parametrize("soft_target_cross_entropy", [standard_soft_target_cross_entropy, barron_soft_target_cross_entropy, new_soft_target_cross_entropy], ids=["standard", "barron", "new"])
    @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
    @example(xs=[0.0, 1.0])
    @settings(max_examples=10_000, deadline=None)
    def test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits(soft_target_cross_entropy, xs):
        x = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
        loss = soft_target_cross_entropy(x, x.detach())
        grad, = torch.autograd.grad(loss, x)
>       assert (grad == 0).all()
E       assert tensor(False)
E        +  where tensor(False) = <built-in method all of Tensor object at 0x113588640>()
E        +    where <built-in method all of Tensor object at 0x113588640> = tensor([-5.5511e-17,  0.0000e+00], dtype=torch.float64) == 0.all
E       Falsifying explicit example: test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits(
E           soft_target_cross_entropy=standard_soft_target_cross_entropy,
E           xs=[0.0, 1.0],
E       )

test_pytorch.py:30: AssertionError
_________________ test_softmax_gradients_sum_to_zero[standard] _________________

softmax = <function standard_softmax at 0x111e30d60>

    @pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
>   @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
                   ^^^^^^^^^^

test_jax.py:15: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
.venv/lib/python3.13/site-packages/hypothesis/core.py:1596: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

softmax = <function standard_softmax at 0x111e30d60>, xs = [0.0, 37.0]

    @pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
    @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
    @example(xs=[0.0, 37.0])
    @example(xs=[1.7976931348623157e308, -9.9792015476736e291])
    @settings(max_examples=10_000, deadline=None)
    def test_softmax_gradients_sum_to_zero(softmax, xs):
        x = jnp.array(xs, dtype=jnp.float64)
        J = jax.jacrev(softmax)(x)
>       assert (jnp.abs(J.sum(axis=1)) <= 1e-12 * jnp.abs(J).sum(axis=1)).all()
E       assert Array(False, dtype=bool)
E        +  where Array(False, dtype=bool) = _all()
E        +    where _all = Array([1.23259516e-32, 8.53304763e-17], dtype=float64) <= (1e-12 * Array([1.70660953e-16, 8.53304763e-17], dtype=float64)).all
E        +      where Array([1.23259516e-32, 8.53304763e-17], dtype=float64) = <PjitFunction of <function abs at 0x111b45300>>(Array([-1.23259516e-32, -8.53304763e-17], dtype=float64))
E        +        where <PjitFunction of <function abs at 0x111b45300>> = jnp.abs
E        +        and   Array([-1.23259516e-32, -8.53304763e-17], dtype=float64) = _sum(axis=1)
E        +          where _sum = Array([[ 8.53304763e-17, -8.53304763e-17],\n       [-8.53304763e-17,  0.00000000e+00]], dtype=float64).sum
E        +      and   Array([1.70660953e-16, 8.53304763e-17], dtype=float64) = _sum(axis=1)
E        +        where _sum = Array([[8.53304763e-17, 8.53304763e-17],\n       [8.53304763e-17, 0.00000000e+00]], dtype=float64).sum
E        +          where Array([[8.53304763e-17, 8.53304763e-17],\n       [8.53304763e-17, 0.00000000e+00]], dtype=float64) = <PjitFunction of <function abs at 0x111b45300>>(Array([[ 8.53304763e-17, -8.53304763e-17],\n       [-8.53304763e-17,  0.00000000e+00]], dtype=float64))
E        +            where <PjitFunction of <function abs at 0x111b45300>> = jnp.abs
E       Falsifying explicit example: test_softmax_gradients_sum_to_zero(
E           softmax=standard_softmax,
E           xs=[0.0, 37.0],
E       )

test_jax.py:22: AssertionError
__________________ test_softmax_gradients_sum_to_zero[barron] __________________

softmax = <function <lambda> at 0x111e30fe0>

    @pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
>   @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
                   ^^^^^^^^^^

test_jax.py:15: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
.venv/lib/python3.13/site-packages/hypothesis/core.py:1596: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

softmax = <function <lambda> at 0x111e30fe0>
xs = [1.7976931348623157e+308, -9.9792015476736e+291]

    @pytest.mark.parametrize("softmax", [standard_softmax, barron_softmax, new_softmax], ids=["standard", "barron", "new"])
    @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
    @example(xs=[0.0, 37.0])
    @example(xs=[1.7976931348623157e308, -9.9792015476736e291])
    @settings(max_examples=10_000, deadline=None)
    def test_softmax_gradients_sum_to_zero(softmax, xs):
        x = jnp.array(xs, dtype=jnp.float64)
        J = jax.jacrev(softmax)(x)
>       assert (jnp.abs(J.sum(axis=1)) <= 1e-12 * jnp.abs(J).sum(axis=1)).all()
E       assert Array(False, dtype=bool)
E        +  where Array(False, dtype=bool) = _all()
E        +    where _all = Array([nan, nan], dtype=float64) <= (1e-12 * Array([nan, nan], dtype=float64)).all
E        +      where Array([nan, nan], dtype=float64) = <PjitFunction of <function abs at 0x111b45300>>(Array([nan, nan], dtype=float64))
E        +        where <PjitFunction of <function abs at 0x111b45300>> = jnp.abs
E        +        and   Array([nan, nan], dtype=float64) = _sum(axis=1)
E        +          where _sum = Array([[nan, nan],\n       [nan, nan]], dtype=float64).sum
E        +      and   Array([nan, nan], dtype=float64) = _sum(axis=1)
E        +        where _sum = Array([[nan, nan],\n       [nan, nan]], dtype=float64).sum
E        +          where Array([[nan, nan],\n       [nan, nan]], dtype=float64) = <PjitFunction of <function abs at 0x111b45300>>(Array([[nan, nan],\n       [nan, nan]], dtype=float64))
E        +            where <PjitFunction of <function abs at 0x111b45300>> = jnp.abs
E       Falsifying explicit example: test_softmax_gradients_sum_to_zero(
E           softmax=lambda z: jnp.exp(barron_log_softmax(z)),
E           xs=[1.7976931348623157e+308, -9.9792015476736e+291],
E       )

test_jax.py:22: AssertionError
_ test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits[standard] _

soft_target_cross_entropy = <function standard_soft_target_cross_entropy at 0x111e30ea0>

    @pytest.mark.parametrize("soft_target_cross_entropy", [standard_soft_target_cross_entropy, barron_soft_target_cross_entropy, new_soft_target_cross_entropy], ids=["standard", "barron", "new"])
>   @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

test_jax.py:26: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
.venv/lib/python3.13/site-packages/hypothesis/core.py:1596: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

soft_target_cross_entropy = <function standard_soft_target_cross_entropy at 0x111e30ea0>
xs = [0.0, 1.0]

    @pytest.mark.parametrize("soft_target_cross_entropy", [standard_soft_target_cross_entropy, barron_soft_target_cross_entropy, new_soft_target_cross_entropy], ids=["standard", "barron", "new"])
    @given(xs=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2))
    @example(xs=[0.0, 1.0])
    @settings(max_examples=10_000, deadline=None)
    def test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits(soft_target_cross_entropy, xs):
        x = jnp.array(xs, dtype=jnp.float64)
        grad = jax.grad(soft_target_cross_entropy, argnums=0)(x, jax.lax.stop_gradient(x))
>       assert jnp.all(grad == 0)
E       assert Array(False, dtype=bool)
E        +  where Array(False, dtype=bool) = <function all at 0x111ac8720>(Array([-8.51911056e-18,  0.00000000e+00], dtype=float64) == 0)
E        +    where <function all at 0x111ac8720> = jnp.all
E       Falsifying explicit example: test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits(
E           soft_target_cross_entropy=standard_soft_target_cross_entropy,
E           xs=[0.0, 1.0],
E       )

test_jax.py:32: AssertionError
=========================== short test summary info ============================
FAILED test_pytorch.py::test_softmax_gradients_sum_to_zero[standard] - assert...
FAILED test_pytorch.py::test_softmax_gradients_sum_to_zero[barron] - assert t...
FAILED test_pytorch.py::test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits[standard]
FAILED test_jax.py::test_softmax_gradients_sum_to_zero[standard] - assert Arr...
FAILED test_jax.py::test_softmax_gradients_sum_to_zero[barron] - assert Array...
FAILED test_jax.py::test_soft_target_cross_entropy_has_zero_gradient_for_identical_logits[standard]
========================= 6 failed, 6 passed in 2.20s ==========================
```
