# orkans
Automate data collection for [pySTEPS](https://pysteps.readthedocs.io/en/latest/index.html) precipitation nowcasting model parameter tuning.
## Description

[Pysteps](https://pysteps.readthedocs.io/en/latest/index.html) is a library for precipitation nowcasting. It contains multiple models and each model has several parameters that the user can change. Although each model works with the default parameters, it is not clear if the default parameters are optimal for different precipitation scenarios.

Manually running tens or hundreds of model parameter combinations for tens or hundreds of precipitation events to evaluate model performance would be extremely tedious. This library aims to automate model execution and data collection such that a large batch of nowcasts can be run in one go, and the results saved in tabular form for later exploration.

## Installation

There is a conda environment configuration file that can be used to setup an environment with all the required dependencies.
Assuming [Anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed:

1. Create an environment using a specific version of Python and the Jupyter kernel to run notebooks. As of writing, `pysteps` does not recommend newer versions of Python. Installing `ipykernel` like this ensures that the notebook kernel has the same Python version as well.

``` conda create -n ENV_NAME python=3.10.9 ipykernel ```

2. Activate the environment

```conda activate ENV_NAME```

3. Install `pysteps`

```conda config --env --prepend channels conda-forge```

```$ conda config --env --set channel_priority strict```

```$ conda install pysteps=1.7.2```

4. Install all the other required libraries from the given configuration file. The `pysteps` library was installed separately because that way the whole process is faster (don't know why). If conda still fails to solve the environment in reasonable time, you can manually install the dependecies listed in `environment-dev.yml` using `conda install` command. 

``` conda env update -n ENV_NAME -f environment-dev.yml ```

5. Verify that `pysteps` works by `import pysteps` in Python. A messege should appear showing where the configuration file `pystepsrc` is located. It should point to the root directory of this library.

## Instructions

TODO
