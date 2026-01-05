from stac_mjx import version


def test_version_is_defined():
    assert isinstance(version.__version__, str)
    assert version.__version__
