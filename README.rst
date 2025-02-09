=================
ml-neural-network
=================


.. image:: https://img.shields.io/pypi/v/ml_neural_network.svg
        :target: https://pypi.python.org/pypi/ml_neural_network

.. image:: https://img.shields.io/travis/newtonsarker/ml_neural_network.svg
        :target: https://travis-ci.com/newtonsarker/ml_neural_network

.. image:: https://readthedocs.org/projects/ml-neural-network/badge/?version=latest
        :target: https://ml-neural-network.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Neural network lab


* Free software: MIT license
* Documentation: https://ml-neural-network.readthedocs.io.


Project Setup
--------

### Create virtual environment
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments

```bash
# create
python3.12 -m venv ./.venv

# activate
source ./.venv/bin/activate

# deactivate
deactivate
```

### Install cookiecutter and create a project template
https://pypi.org/project/cookiecutter/
```bash
# install pipx
pip install pipx

# install cookiecutter
pipx install cookiecutter

# create project template
pipx run cookiecutter gh:audreyfeldroy/cookiecutter-pypackage
```

### Install dependencies
```bash
pip install -r requirements_dev.txt
```
