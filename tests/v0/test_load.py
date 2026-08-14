def test_load_priormean(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_PriorMean_H, load_PriorMean_phi

    load_PriorMean_H()
    load_PriorMean_phi()


def test_load_GPR(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_GPR_H, load_GPR_phi

    load_GPR_H()
    load_GPR_phi()


def test_load_GPQR(models_path, monkeypatch):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_GPQR_H, load_GPQR_phi

    load_GPQR_H()
    load_GPQR_phi()
