-include ../config/do.mk

DO_what=      AGENT: multi-objective analysis 
DO_copyright= Copyright (c) 2023 Tim Menzies, BSD-2.
DO_repos=     . ../config ../data

install: ## load python3 packages (requires `pip3`)
	 pip3 install -qr requirements.txt

../data:
	(cd ..; git clone https://gist.github.com/d47b8699d9953eef14d516d6e54e742e.git data)

../config:
	(cd ..; git clone https://gist.github.com/42f78b8beec9e98434b55438f9983ecc.git config)

tests: ## run test suite
	PYTHONDONTWRITEBYTECODE=1 pytest -q tests.py

doc: ## generate documentation
	pdoc --show-source \
		   --logo 'https://hetmanrecovery.com/pic/out/hetman_internet_spy_256x256.png' *.py


