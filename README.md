# rrm-slice-rl

Code containing RRM simulation using RL in a scenario with RAN slicing.

## Install

- Install [pipenv](https://github.com/pypa/pipenv)
- Install dependencies using pipenv: `pipenv install`
- To access the virtual environment created, run `pipenv shell`, now all commands which you run will be performed into virtual enviroment created
- (In case you want to contribute with this repo, if not you can skip this step) Activate pre-commit hooks to use [black formatter](https://github.com/psf/black), [flake8 lint](https://gitlab.com/pycqa/flake8) and [Isort references](https://github.com/timothycrosley/isort). Run `pre-commit install`. Now every time you make a commit, black formatter, flake8 and isort will make tests to verify if your code is following the [patterns](https://realpython.com/python-pep8/) (you can adapt your IDE or text editor to follow this patterns, e.g. [vs code](https://code.visualstudio.com/docs/python/python-tutorial#_next-steps))

## Hyperparameters optimization using Optuna

Run the script `optimize.py` using pipenv running `pipenv run python optimize.py`. It would take a long time to Optuna generate the optimized hyperparameters for each scenario, so you can use the hyperparameters that it were already generated into `hyperparameter_opt` folder and **skip this step**.
## Training and testing ML model 

Run the command `pipenv run python run.py` to start the simulation for all scenarios. You can watch the training performance using tensorboard. After the training finish, the RL models will be saved into `agents` folder, the VecNormalize parameters into `vecnormalize_models` folder, the evaluations made along with the trainign into `evaluations` folder, and the test results into `hist` folder.
## Generating figures with the results

To generate the figures with results obtained in the paper, you can run `pipenv run python plot_results.py` and the figures should be available into the folder `results" as soon as the script finish.

## Citing this project
To cite this repository in publications:
```
@ARTICLE{nahum2023rrs,
  author={Nahum, Cleverson V. and Lopes, Victor Hugo and Dreifuerst, Ryan M. and Batista, Pedro and Correa, Ilan and Cardoso, Kleber V. and Klautau, Aldebaro and Heath, Robert W.},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Intent-aware Radio Resource Scheduling in a RAN Slicing Scenario using Reinforcement Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TWC.2023.3297014}}
```
