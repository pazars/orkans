import pytest
import yaml

from orkans import nowcast
from orkans import TEST_CFG_DIR


def get_test_config():
    cfg_abs_path = (TEST_CFG_DIR / "config.yaml").resolve()
    with open(cfg_abs_path.as_posix(), "r") as file:
        return yaml.safe_load(file)


def check_results(res, ref, thr):
    for key, ref_value in ref.items():
        res_value = res[key]
        if type(ref_value) == str:
            assert str(res_value) == str(ref_value)
        elif type(ref_value) == float:
            assert abs(res_value - ref_value) / ref_value < thr


def test_pysteps_steps():

    cfg = get_test_config()
    res = nowcast.run("steps", cfg, test=True)

    ref = {
        "seed": 2023,
        "id": "2f50d60198e3",
        "precip_ratio": 0.31913967258794845,
        "data_date": "201808241915",
        "max_rrate": 39.65,
        "mean_rrate_above_thr": 0.9335070486584811,
        "precip_above_2mmh": 0.12772472335910262,
        "precip_above_5mmh": 0.03192360163710777,
        "precip_above_8mmh": 0.012581476428679702,
        "CSI_T15_THR0.1": 0.6607770716284119,
        "MAE_T15_THR0.1": 0.30352391800484807,
        "RMSE_T15_THR0.1": 1.9827727738211145,
        "RV_T15_THR0.1": -1.916641073941334,
        "roc_area_T15_THR0.1": 0.9234563362884807,
    }

    # Check that results are within threshold
    # e.g. threshold 0.01 means results within 1%
    check_results(res, ref, 0.01)


def test_pysteps_anvil():

    cfg = get_test_config()
    res = nowcast.run("anvil", cfg, test=True)

    ref = {
        "id": "7b01554447f0",
        "precip_ratio": 0.31913967258794845,
        "data_date": "201808241915",
        "max_rrate": 39.65,
        "mean_rrate_above_thr": 0.9335070486584811,
        "precip_above_2mmh": 0.12772472335910262,
        "precip_above_5mmh": 0.03192360163710777,
        "precip_above_8mmh": 0.012581476428679702,
        "MAE_T15_THR0.1": 0.30972514607954293,
        "RMSE_T15_THR0.1": 2.4914395186052904,
        "RV_T15_THR0.1": -3.549664357782895,
    }

    # Check that results are within threshold
    # e.g. threshold 0.01 means results within 1%
    check_results(res, ref, 0.01)


def test_pysteps_sseps():

    cfg = get_test_config()
    res = nowcast.run("sseps", cfg, test=True)

    ref = {
        "seed": 2023,
        "id": "516c5f07b89f",
        "precip_ratio": 0.31913967258794845,
        "data_date": "201808241915",
        "max_rrate": 39.65,
        "mean_rrate_above_thr": 0.9335070486584811,
        "precip_above_2mmh": 0.12772472335910262,
        "precip_above_5mmh": 0.03192360163710777,
        "precip_above_8mmh": 0.012581476428679702,
        "CSI_T15_THR0.1": 0.6683089366528675,
        "MAE_T15_THR0.1": 0.29823391663607285,
        "RMSE_T15_THR0.1": 1.9993673648979022,
        "RV_T15_THR0.1": -1.9645749027631718,
        "roc_area_T15_THR0.1": 0.9270941859711961,
    }

    # Check that results are within threshold
    # e.g. threshold 0.01 means results within 1%
    check_results(res, ref, 0.01)


def test_pysteps_linda():

    cfg = get_test_config()
    res = nowcast.run("linda", cfg, test=True)

    ref = {
        "seed": 2023,
        "id": "1b9260268f64",
        "precip_ratio": 0.31913967258794845,
        "data_date": "201808241915",
        "max_rrate": 39.65,
        "mean_rrate_above_thr": 0.9335070486584811,
        "precip_above_2mmh": 0.12772472335910262,
        "precip_above_5mmh": 0.03192360163710777,
        "precip_above_8mmh": 0.012581476428679702,
        "MAE_T15_THR0.1": 0.29643854448337453,
        "RMSE_T15_THR0.1": 1.7367547152334804,
        "RV_T15_THR0.1": -1.2108314514333167,
    }

    # Check that results are within threshold
    # e.g. threshold 0.01 means results within 1%
    check_results(res, ref, 0.01)
