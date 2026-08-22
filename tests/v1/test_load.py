def test_load_priormean(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_PriorMean

    model = load_PriorMean()
    assert model.output_names == ("H", "phi_1", "phi_3")


def test_load_gpr(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_GPR

    _, _, _, model = load_GPR()
    assert model.batch_shape[-1] == 3


def test_load_gpqr(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_GPQR

    quantiles, _, _, _, model = load_GPQR()
    assert model.batch_shape[-1] == 3
    assert quantiles.ndim == 1
