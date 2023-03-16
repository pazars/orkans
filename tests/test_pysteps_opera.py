import sys
import pytest

from pathlib import Path

try:
    from orkans import nowcast
except ModuleNotFoundError:
    LIB_DIR = (Path(".") / "..").resolve().as_posix()
    sys.path.append(LIB_DIR)
    from orkans import nowcast


def get_test_config_path():
    cfg_rel_path = Path("tests") / "_test_configs" / "config.yaml"
    cfg_abs_path = cfg_rel_path.resolve()
    return cfg_abs_path.as_posix()


def test_pysteps_steps():

    cfg_path = get_test_config_path()
    res = nowcast.run("steps", cfg_path)

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

    for key, value in ref.items():
        if type(value) == str:
            assert str(res[key]) == str(value)
        elif type(value) == float:
            assert abs(res[key] - value) < 1e-3


def test_pysteps_anvil():

    cfg_path = get_test_config_path()
    res = nowcast.run("anvil", cfg_path)

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

    for key, value in ref.items():
        if type(value) == str:
            assert str(res[key]) == str(value)
        elif type(value) == float:
            assert abs(res[key] - value) < 1e-3


def test_pysteps_sseps():

    cfg_path = get_test_config_path()
    res = nowcast.run("sseps", cfg_path)

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

    for key, value in ref.items():
        if type(value) == str:
            assert str(res[key]) == str(value)
        elif type(value) == float:
            assert abs(res[key] - value) < 1e-3


def test_pysteps_linda():

    cfg_path = get_test_config_path()
    res = nowcast.run("linda", cfg_path)

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

    for key, value in ref.items():
        if type(value) == str:
            assert str(res[key]) == str(value)
        elif type(value) == float:
            assert abs(res[key] - value) < 1e-3
