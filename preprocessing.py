import numpy as np


class PreProcessor:
    def __init__(self) -> None:
        self.data = None
        self.metadata = None

    def add_data(self, data: np.ndarray, metadata: dict) -> None:
        self.data = data
        self.metadata = metadata

    def precipitation_ratio(self) -> bool:

        last_data = self.data[-1, :, :]
        last_data_no_nan = last_data[~np.isnan(last_data)]
        n_percip_pix = last_data_no_nan[last_data_no_nan > self.metadata["threshold"]].size
        percip_ratio = n_percip_pix / last_data_no_nan.size
        return percip_ratio
