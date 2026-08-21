"""MPI ownership bootstrap for ImpactX.

``ImpactX.finalize()`` calls ``amrex::Finalize()``, which calls ``MPI_Finalize()``
when AMReX was the one that initialized MPI. ``MPI_Finalize`` is terminal: any later
``ImpactX()`` / ``init_grids()`` aborts with

    Attempting to use an MPI routine before initializing or after finalizing MPICH

That matters here because :class:`lume_impactx.simulator.ImpactXSimulator` rebuilds the
simulation on every ``track()``, so it finalizes many times in one process.

Importing ``mpi4py`` first makes *mpi4py* the owner of ``MPI_Init``/``MPI_Finalize``.
AMReX then sees MPI already initialized and leaves it alone, so ``finalize()`` becomes
safe to call repeatedly. This is the same trick ImpactX uses in its own test suite
(``tests/python/conftest.py``).

Measured with an MPI-enabled ImpactX 26.06: 200 build/track/finalize cycles at ~6 ms
each, bit-identical results. Without this bootstrap the process aborts on cycle 2.
"""

from __future__ import annotations

_bootstrapped = False


def ensure_external_mpi() -> bool:
    """Hand MPI ownership to ``mpi4py`` if this ImpactX build uses MPI.

    Idempotent and safe to call when ImpactX is a serial build or when ``mpi4py`` is
    not installed.

    Returns
    -------
    bool
        True if ``mpi4py`` now owns MPI, False if there was nothing to do.
    """
    global _bootstrapped
    if _bootstrapped:
        return True

    try:
        import impactx
    except ImportError:
        return False

    if not impactx.Config.have_mpi:
        _bootstrapped = True
        return False

    try:
        from mpi4py import MPI  # noqa: F401  -- import takes ownership of MPI_Init
    except ImportError as exc:  # pragma: no cover - depends on the local build
        raise ImportError(
            "This ImpactX build uses MPI, so lume-impactx needs mpi4py to own "
            "MPI_Init/MPI_Finalize. Without it, the second simulation in a process "
            "aborts. Install mpi4py, or use a serial ImpactX build (impactx-noacc)."
        ) from exc

    _bootstrapped = True
    return True
