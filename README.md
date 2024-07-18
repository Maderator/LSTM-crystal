# LSTM-crystal-growth

This repository contains the code accompanying the Master thesis 
"Study of LSTM Networks on Natural Sciences Data". It includes the Python implementation of the LSTM networks with the cell-level and architecture level variations together with experiments on the VGF GaAs growth simulation dataset.  
The repository also contains Matlab implementation of LSTM networks with peephole and residual connections. After the comparison of Matlab and Tensorflow implementations, the Matlab implementation was not further developed.

## How to obtain dataset
- Feel free to contact me at `janmadera97+mff_thesis_data at gmail.com` and I will ask for premission to share the dataset with you.
- If the email address is not available, please, try it again without the `+mff_thesis_data` part.
- After obtaining the dataset, name it as `sequencesInputsOutputs.mat` and save it to the `data/` folder.
- Afterward, run the `python/data_processing/train_test_split.ipynb` notebook to prepare data splits for training and testing.

## Repository structure

The repository is structured as follows:
- [data/](data/) - contains the VGF GaAs growth simulation raw dataset (not included in the repository and needs to be obtained separately)
- [experiments/](experiments/) - contains the experiments with LSTM networks on the VGF GaAs growth simulation dataset
- [matlab/](matlab/) - contains the drafts of Matlab implementation of LSTM networks with peephole and residual connections
- [python/](python/) - contains the Python implementation of LSTM networks with cell-level and architecture level variations
  - The custom LSTM cell and architecture implementations are located in the python/predictor/models directory
  - The [rnn_cell.py](python/predictor/models/rnn_cell.py) file contains the implementation of the custom LSTM cell with all the variations like peephole connections, residual connections, layer normalization, and there are also implementations of LSTM cells with only residual connections or only peephole connections.
  - The [custom_layers.py](pyhton/predictor/models/custom_layers.py) file contains the implementation of the DilatedRNN layer described in our thesis. 
  - The [rnn_constructor.py](python/predictor/models/rnn_constructor.py) file contains a parser of the parameters (from basic_parameters.py file) and it construct the multi layered LSTM networks based on the received parameters.
- [README.MD](README.md) - this file

## Installation
Project dependencies have to be installed before the experiments can be run.
The project was implemented on Windows 11 and therefore if the project is run on
different OS, we cannot guarantee that it will work properly.

### Windows

#### 1. Python installation
#### 1. Install pyenv-win
- See [Quick start](https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#quick-start) for installation guide.

#### 1. Install Python 3.10.11, pip, and all dependencies
We used Python version 3.10.11 as the newer versions were not compatible with 
some dependencies. 

1. To install Python 3.10.11 use package installer available online at adress https://www.python.org/downloads/release/python-31011/
- Add python to Windows PATH variable. 
  - If the option "Add Python X.Y to PATH" is not selected, the install directory can be added to PATH variable later by following one of many tutorials available online (for example this one https://realpython.com/add-python-to-path/).
  - Check if the Python is available (and which version) in Powershell using command
    ```Powershell
    python --version
    ```
2. Pip should be installed along with python (try command `pip --version` in powershell to see which version of pip is installed). If it is not installed, try calling this command in powershell: 
    ```Powershell
    python -m ensurepip --upgrade
    ```

3. Install setuptools and wheel packages using pip:
    ```Powershell
    pip install --upgrade setuptools wheel
    ```

- Optional step is to install pyenv environment or anaconda to manage python environments.

4. Install all project dependencies using pip:
   - In the root directory of the project, call this command:
    ```Powershell
    pip install -r requirements.txt
    ```

5. Install the project as a python package:
   1. In order to use our models which are implemented in `python/predictor` directory and other functions/classes implemented in the `python/` or `experiments/` directory from anywhere in the project, we need first to install them as a python package:
   - In the root directory of the project, call this command:
        ```Powershell
        pip install -e .
        ```
        The -e option (or --editable) installs project in editable mode so that new functions/classes and changes made to file in any package of the project (folders containing file `__init__.py`) can be used without first having to install the project again.

## Running the experiments
- The file [python/basic_parameters.py](python/basic_parameters.py) contains the basic parameters for the experiments. See this file for inspiration for parameters to change and test in the experiments.
- If the dataset is obtained, saved to the `data/` folder, and the `python/data_processing/train_test_split.ipynb` notebook was successfuly run, the experiments can be run.
- Simply copy folder of the experiments or choose one of the already existing folders, open one of the .py files, change the parameters if needed, change name of the file as the logs are saved to the file with the same name in logs subfolder, and run it.

- [experiments/](experiments/) - contains the experiments with LSTM networks on the VGF GaAs growth simulation dataset

- [experiments/00_data_visualization](experiments/00_data_visualization) - contains the notebook for dataset visualization
- [experiments/01_speed_comparison](experiments/01_speed_comparison) - contains the notebook for speed comparison of TensorFlow and the Matlab implementation of LSTM networks with peephole connections or residual connections (Kim et al.)
- [experiments/02_data_preprocessing](experiments/02_data_preprocessing) - contains the notebook for data preprocessing methods comparison
- [experiments/03_optimizers](experiments/03_optimizers) - contains the notebook for optimizers comparison
- [experiments/04_gradient_clipping](experiments/04_gradient_clipping) - contains the notebook for gradient clipping methods comparison
- [experiments/05_normalization_dropout](experiments/05_normalization_dropout) - contains the notebook for normalization and dropout methods comparison
- [experiments/07_single_layer_rnns](experiments/07_single_layer_rnns) - contains the notebook for single-layer RNNs comparison
- [experiments/08_multi_layer_comparison](experiments/08_multi_layer_comparison) - contains the notebook for multi-layer RNNs comparison
- [experiments/09_test_dataset_eval](experiments/09_test_dataset_eval) - contains the notebook for evaluation of the test dataset