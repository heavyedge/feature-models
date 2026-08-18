def test_load_priormean(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_PriorMean

    load_PriorMean()


def test_load_gpr(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_GPR

    load_GPR()


def test_load_gpqr(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_GPQR

    load_GPQR()
