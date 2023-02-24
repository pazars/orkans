import numpy as np

from datetime import datetime


class PreProcessor:
    def __init__(self) -> None:
        self.data = None
        self.metadata = None
        self._unit = "mm/h"

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value: str):
        if self.data:
            raise RuntimeError("Can't change unit when data already set.")
        self._unit = value

    def _get_data_step(self, idx: int, exclude_nan: bool = False) -> np.ndarray:
        """Get data array at a specified timestep.

        Args:
            idx (int): Timestep index
            exclude_nan (bool, optional): Exclude NaN values. Defaults to False.

        Returns:
            numpy.ndarray: Data array.
        """
        data = self.data[idx, :, :]
        if exclude_nan:
            return data[~np.isnan(data)]
        return data

    def add_data(self, data: np.ndarray, metadata: dict) -> None:
        """Add data and its metadata imported using pysteps.

        Args:
            data (np.ndarray): Data array
            metadata (dict): Metadata
        """
        self.data = data
        self.metadata = metadata

        # Check if data is in correct units
        unit = self.metadata["unit"]
        if not unit == self._unit:
            msg = f"Data unit should be in '{self._unit}', but given in '{unit}'"
            raise RuntimeError(msg)

    def precipitation_ratio(self, idx: int) -> float:
        """Calculate ratio of points with non-zero precipitation values in data[idx].

        Args:
            idx (int): Data timestep index.

        Returns:
            float: Precipitation ratio (0-1).
        """

        data = self._get_data_step(-1, True)
        n_percip_pix = data[data > self.metadata["threshold"]].size
        percip_ratio = n_percip_pix / data.size
        return percip_ratio

    def collect_info(self) -> dict:
        """Collect information about the observation data for output.

        Returns:
            dict: Observation information.
        """

        info = {}

        tstep_idx = -1
        data = self._get_data_step(tstep_idx, True)
        datetime_obj = self.metadata["timestamps"][tstep_idx]

        # Observation date
        info["date"] = datetime.strftime(datetime_obj, "%Y%m%d%H%M")

        # Maximum precipitation rate
        info["max_rrate"] = data.max()

        # Mean precipitation rate of values above threshold
        data_above_thr = data[data > self.metadata["threshold"]]
        info["mean_rrate_above_thr"] = data_above_thr.mean()

        return info
