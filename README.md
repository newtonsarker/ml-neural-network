# ml-neural-network

## Create virtual environment
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments
```bash
# create
python3.12 -m venv ./.venv

# activate
source ./.venv/bin/activate

# deactivate
deactivate
```

## Install cookiecutter and create a project template
https://pypi.org/project/cookiecutter/
```bash
# install pipx
pip install pipx

# install cookiecutter
pipx install cookiecutter

# create project template
pipx run cookiecutter gh:audreyfeldroy/cookiecutter-pypackage
```

## Install dependencies
```bash
pip install -r requirements_dev.txt
```