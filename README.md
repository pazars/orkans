# orkans
[Pysteps](https://pysteps.readthedocs.io/en/latest/index.html) automation and result collection for model parameter tuning.

## Description

[Pysteps](https://pysteps.readthedocs.io/en/latest/index.html) is a library for precipitation nowcasting. It contains multiple models and each model has several parameters that the user can change. Although each model works with the default parameters, it is not clear if the default parameters are optimal for different precipitation scenarios.

Manually running tens or hundreds of model parameter combinations for tens or hundreds of precipitation events to evaluate model performance would be extremely tedious. This library aims to automate model execution and data collection such that a large batch of nowcasts can be run in one go, and the results saved in tabular form for later exploration.

## Installation

There is a conda environment configuration file that can be used to setup an environment with all the required dependencies.
Assuming [Anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed, run:

``` conda env create -f environment.yml ```

This will create an environment called `nwc` (short for nowcast). If you would like to change the name, edit the first line of `environment.yml`.

## Instructions

TODO
