import numpy as np

from datetime import datetime
from pysteps.utils import transformation, to_rainrate
from scipy import stats


class PreProcessor:
    """Class for pre-processing pysteps data."""

    def __init__(self, data: np.ndarray, metadata: dict) -> None:
        """Initialize data and check if units are correct.

        Args:
            data (np.ndarray): Data from pysteps importer
            metadata (dict): Metadata from pysteps importer
        """
        self.data = data
        self.metadata = metadata
        self._unit = "mm/h"

        # Check if data is in correct units
        unit = self.metadata["unit"]
        if not unit == self._unit:
            # TODO: Add warning that converting units to rainrate
            # Need to get logger to here.
            self.data, self.metadata = to_rainrate(self.data, self.metadata)

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

    def precipitation_ratio(self) -> float:
        """Calculate ratio of points with non-zero precipitation at last timestep.

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
        info["data_date"] = datetime.strftime(datetime_obj, "%Y%m%d%H%M")

        # Maximum precipitation rate
        info["max_rrate"] = data.max()

        # Mean precipitation rate of values above threshold
        data_above_thr = data[data > self.metadata["threshold"]]
        info["mean_rrate_above_thr"] = data_above_thr.mean()

        # Ratio of pixels with rain rate values
        info["precip_ratio"] = self.precipitation_ratio()

        return info

    def find_best_transformation(self, verbose: bool = False) -> tuple:
        """Find most suitable data transformation function for velocity field estimation.

        Args:
            verbose (bool, optional): Print information. Defaults to False.

        Returns:
            tuple[function, float | None]: Transformation function and its argument, if applicable.
        """

        rrate = self.data
        mdata = self.metadata

        if verbose:
            print("-- Finding best data transformation --")

        skew_threshold = 0.2
        best_mean = 1e6
        best_skew = None

        best_transform = None
        extra_arg = None

        # Map of transformation methods
        transforms = {
            "dB": transformation.dB_transform,
            "sqrt": transformation.sqrt_transform,
            "box-cox": transformation.boxcox_transform,
            "nq": transformation.NQ_transform,
        }

        # Exclude zero values as most transformations can't deal with them
        rrate = rrate[rrate > mdata["zerovalue"]]

        # Reduce data array dimensionality to 1D for stats functions
        rrate_flat = rrate.flatten()

        for name, func in transforms.items():

            # Apply transformation. Extra step for Box-Cox
            if name != "box-cox":
                rrate_, _ = func(rrate_flat, mdata)
            else:
                Lambda = find_best_boxcox_lambda(rrate_flat, mdata, verbose=verbose)
                rrate_, _ = func(rrate_flat, mdata, Lambda)

            skewness = stats.skew(rrate_)

            # Skip if skewness doesn't meet threshold
            if skewness > skew_threshold:
                continue

            # Skip if no mean value improvement
            mean = np.mean(rrate_)
            if abs(mean) > abs(best_mean):
                continue

            # Update best performing method variables
            best_skew = skewness
            best_mean = mean
            best_transform = name

            if name == "box-cox":
                extra_arg = Lambda

        if verbose:
            print("Best transform method:", best_transform)
            print("Skewness:", best_skew)
            print(f"Mean: {best_mean}\n")

        # Apply best transformation to data with non-reduced dimensionality
        best_func = transforms[best_transform]
        # best_data, best_metadata = best_func(rrate, mdata)

        return (best_func, extra_arg)

    def apply_transformation(self, tfunc, arg: float = None) -> tuple[np.ndarray, dict]:
        """Apply transformation function to pysteps data.

        Function argument (lambda) only applicable for Box-Cox transform.

        Args:
            tfunc (function): Transformation function
            arg (float, optional): Function argument. Defaults to None.

        Returns:
            tuple[np.ndarray, dict]: Transformed data and its metadata.
        """

        # Even though the transformation was already applied in the find function,
        # it is applied again, because previously the data was flattened to 1D.
        # This approach avoids copying the data in the find function.
        if arg:
            return tfunc(self.data, self.metadata, Lambda=arg)
        return tfunc(self.data, self.metadata)


def find_best_boxcox_lambda(data: np.ndarray, mdata: dict, verbose: bool = False) -> float:
    """Find best lambda parameter for Box-Cox transformation function.

    Args:
        data (np.ndarray): Data from pysteps
        mdata (dict): Data metadata
        verbose (bool, optional): Print information. Defaults to False.

    Returns:
        float: Best lambda parameter
    """

    data = []
    labels = []
    skw = []

    # Keep only positive rainfall values
    data_flat = data[data > mdata["zerovalue"]].flatten()

    # Test a range of values for the transformation parameter Lambda
    lambdas = np.linspace(-0.4, 0.4, 11)
    for Lambda in lambdas:
        R_, _ = transformation.boxcox_transform(data_flat, mdata, Lambda)
        R_ = (R_ - np.mean(R_)) / np.std(R_)
        data.append(R_)
        labels.append("{0:.2f}".format(Lambda))
        skw.append(stats.skew(R_))  # skewness

    # Best lambda
    idx_best = np.argmin(np.abs(skw))

    if verbose:
        print("Best Box-Cox lambda:", lambdas[idx_best])
        print(f"Best Box-Cox skewness={skw[idx_best]} for lambda={lambdas[idx_best]}\n")

    return lambdas[idx_best]
