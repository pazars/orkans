# Orkans
Automate data collection for [pySTEPS](https://pysteps.readthedocs.io/en/latest/index.html) precipitation nowcasting model parameter tuning.
## What is it?

[Pysteps](https://pysteps.readthedocs.io/en/latest/index.html) is a library for precipitation nowcasting. It contains multiple models and each model has several parameters that the user can change. Although each model works with the default parameters, it is not clear if the default parameters are optimal for different precipitation scenarios.

Manually running tens or hundreds of model parameter combinations for tens or hundreds of precipitation events to evaluate model performance would be extremely tedious. This library aims to automate model execution and data collection such that a large batch of nowcasts can be run in one go, and the results saved in tabular form for later exploration.

## How to install it?

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

## How to use it?

### Running a batch in parallel from Command Line (CL)

The most simple way to run a nowcast is by running the `main.py` file from command line and passing the model name as an argument:

```python main.py -m anvil```

Here, `-m` stands for model and possible choices are *anvil*, *steps*, *sseps*, *linda*. It is also possible to use `--model`. The main script will load the `config.yaml` configuration file that contains information about where to find radar data, for which datetimes nowcasts should be run and with how many timesteps etc. It also specifies all the
default parameters that will be used by the model. All of this information can be changed.

The model name is specified separately in the CL in order to allow batches of multiple models to run in parallel. For example, if one would like to run *anvil* and *steps* in parellel:

```python main.py -m anvil steps -n 2```

The `-n` or alternatively `--nproc` specifies the number of parallel processes to create. The default value is 1.

Another use case running batches of model parameters:

```python main.py -m steps --batch```

The range of parameters that is used in a batch run is also specified in `config.yaml` and can be changed by the user. Each model will run by changing one parameter at a time from the default values. This means that the impact of a single parameter value is evaluated instead of finding the best combination out of all parameters!

Combining all of this, a multi-model parallel batch run could look like this:

```python main.py -m steps sseps --batch -n 4```

One can always check information about the available parameters using:

```python main.py --help```

### Running a single nowcast (mostly for debugging)

While CL runs offer flexibility and speed, if something goes wrong, it's better to run the code directly in order to use a debugger. In that case one should use the `./orkans/nowcast.py` file. This is what `main.py` calls once the input parameters are parsed. It will also read the configuration file `config.yaml` and use `steps` as the default nowcast model. One can change this at the end of the `nowcast.py` code.

## How to interpret the results?

All results are stored in the `./results` directory. If missing, the directory is automatically created at the end of a run. It will contain a `nowcasts_MODEL_NAME.csv` for each model, a `plots` directory and a `netcdf` directory if netcdf export is enabled in `config.yaml`.

The csv file contains the run id, some pre-processing info about the data, model parameters as well as the evaluated metrics. For each run id, there is a sub-directory in `plots` containing various plots like ROC curves, reliability diagrams, rank histograms etc. The exact output depends whether the model is determinstic or probabilistic. 

**Important note:** It can happen that the run ids are not unique. What I like to do limit the possibility of this happening is run small batches and then rename the results to something else e.g. `results_steps_2022`.

## How is it implemented?

So far we looked at:

- `./main.py`: Convient for running multiple models from CL
- `./orkans/nowcast.py`: The actual nowcast workflow

However, there are other scripts one might be interested in looking at/modiying:

- `./orkans/preprocessing.py`: Calculates some metrics about the input data like the ratio of pixels containing precipitation and also finds and applies the best data transformation method.
- `./orkans/postprocessing.py`: Contains classes for saving plots and evaluating model metrics. If one wants to modify what plots are saved, in what format, for what timesteps and so on, this is the place to do that. 

Other files that one probably won't need to modify:

- `./events.py`: Checks existing runs in the results in order to avoid running nowcasts that already have results. 
- `./orkans/utils.py`: Some utility functions. Mainly wrappers for loading files to improve readability in `nowcast.py`.
- `./etc/clidata.py`: Example code for accessing CLIDATA. Just an example, not actually used anywhere in the code.
- `./etc/fetch_pysteps_data.py`: Script for downloading `pysteps` test data and default `pystepsrc` configuration file. Only relevant for developing and testing new features. Used this in the beginning when OPERA data was not available.
- Various notebooks in `./notebooks` for input/output data exploration and so on.