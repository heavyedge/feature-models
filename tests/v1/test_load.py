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

    quantiles, lower_bound, _, _, likelihood, model = load_GPQR()
    assert model.batch_shape[-1] == 3
    assert quantiles.ndim == 1
    assert lower_bound > 0
    assert model.quantile_level_offsets.equal(
        quantiles - quantiles[model.central_quantile_idx[0]]
    )
    assert likelihood.quantile_level_offsets.equal(
        quantiles - quantiles[model.central_quantile_idx[0]]
    )


def test_save_gpqr_preserves_quantile_gap_lower_bound(
    models_path, monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(models_path))
    from feature_models.load import load_GPQR

    from scripts.v1.train.save import save_gpqr

    quantiles, lower_bound, X_scaler, y_scaler, likelihood, model = load_GPQR()
    checkpoint_path = tmp_path / "gpqr.pt"
    save_gpqr(
        quantiles,
        X_scaler,
        y_scaler,
        likelihood,
        model,
        checkpoint_path,
        quantile_gap_lower_bound=lower_bound,
    )

    loaded_quantiles, loaded_lower_bound, *_ = load_GPQR(checkpoint_path)
    assert loaded_quantiles.equal(quantiles)
    assert loaded_lower_bound == lower_bound
