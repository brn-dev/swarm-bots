import inspect
import subprocess
import sys
import textwrap

import pytest


def test_kernel_bindings_resolve_without_python_closure_support() -> None:
    pytest.importorskip("triton")
    from swarmbots.learn.action_dists.gsde_noise import _masked_normal_kernel

    function = _masked_normal_kernel().fn
    bindings = inspect.getclosurevars(function)
    # Triton 3.2/3.3 compile against globals without merging captured locals.
    for name, value in (bindings.globals | bindings.nonlocals).items():
        assert name in function.__globals__, f"Kernel name {name!r} requires unsupported closure capture"
        assert function.__globals__[name] is value


def test_cpu_noise_resets_do_not_import_or_probe_triton() -> None:
    script = textwrap.dedent("""
        import builtins
        import importlib.util
        from unittest.mock import patch

        import torch

        torch.set_num_threads(1)
        original_import = builtins.__import__
        original_find_spec = importlib.util.find_spec

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "triton" or name.startswith("triton."):
                raise AssertionError("CPU noise must not import Triton")
            return original_import(name, *args, **kwargs)

        def guarded_find_spec(name: str, *args: object, **kwargs: object) -> object:
            if name == "triton" or name.startswith("triton."):
                raise AssertionError("CPU noise must not probe Triton")
            return original_find_spec(name, *args, **kwargs)

        with patch.object(builtins, "__import__", guarded_import), patch.object(
            importlib.util, "find_spec", guarded_find_spec,
        ):
            from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist

            distribution = GSDEActionDist(7, 3, .5).cpu()
            distribution.reset_noise((2, 5))
            original = distribution.get_temporal_correlation_state().clone()
            distribution.reset_noise_masked(torch.tensor([True, False]))
            updated = distribution.get_temporal_correlation_state()
            assert torch.equal(updated[1], original[1])
            assert not torch.equal(updated[0], original[0])
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
