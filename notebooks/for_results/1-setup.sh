# assumes that python is installed. Things worked out for Python 3.13.3 on our side,
# they probably work for other python versions, especially newer ones, but they might
# also break for other python versions.
#
# to be safe, we recommend using python 3.13.3, which may be achieved by installing
# pyenv and running the command pyenv local 3.13.3 on the terminal in this directory.
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt