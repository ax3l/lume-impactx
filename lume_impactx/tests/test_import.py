import lume_impactx


def test_package_exposes_version():
    assert isinstance(lume_impactx.__version__, str)
    assert lume_impactx.__version__


def test_mpi_bootstrap_is_idempotent():
    # Safe to call repeatedly, and safe when ImpactX is absent or serial.
    first = lume_impactx.ensure_external_mpi()
    assert lume_impactx.ensure_external_mpi() == first
