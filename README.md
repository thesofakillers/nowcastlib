# Nowcast Library

🧙‍♂️🔧 Utils that can be reused and shared across and beyond the ESO Nowcast
project

This is a public repository hosted on GitHub via a push mirror setup in the
internal ESO GitLab

## Installation

Simply run

```console
pip install nowcastlib
```

## Usage and Documentation

At the moment, Nowcast Library is simply a collection of functions dealing with
raw time series data surrounding the ESO Nowcast project. These functions are in
the module [`rawdata.py`](./nowcastlib/rawdata.py), and can be imported as such

```python
"""Example showing how to access compute_trig_fields function"""
import nowcastlib as ncl
import pandas as pd
import numpy as np

data_df = pd.DataFrame(
    [[0, 3, 4, np.NaN], [32, 4, np.NaN, 4], [56, 8, 0, np.NaN]],
    columns=["A", "B", "C"],
    index=pd.date_range(start="1/1/2018", periods=4, freq="2min"),
)

result = ncl.rawdata.compute_trig_fields(data_df, ["A", "C"])
```

Further documentation [here](https://giuliostarace.com/nowcastlib).

## Development Setup

This repository relies on [Poetry](https://python-poetry.org/) for tracking
dependencies, building and publishing. It is therefore recommended that
developers [install poetry](https://python-poetry.org/docs/#installation) and
make use of it throughout their development of the project.

### Dependencies

Make sure you are in the right Python environment and run

```console
poetry install
```

This reads [pyproject.toml](./pyproject.toml), resolves the dependencies, and
installs them.

### Deployment

The repository is published to [PyPi](https://pypi.org/), so to make it
accessible via a `pip install` command as mentioned [earlier](#install).

To publish changes follow these steps:

1. Changes should be merged into the master branch. Ideally this process is
   automated via a CI tool.
2. Optionally run
   [`poetry version`](https://python-poetry.org/docs/cli/#version) with the
   appropriate argument based on [semver guidelines](https://semver.org/).
3. Prepare the package by running

   ```console
   poetry build
   ```

4. Ensure you have [TestPyPi](https://test.pypi.org/) and PyPi configured as
   your poetry repositories:

   ```console
   > poetry config repositories.testpypi https://test.pypi.org/legacy/

   > poetry config repositories.pypi https://pypi.org/

   > poetry config --list
   repositories.pypi.url = "https://pypi.org/"
   repositories.testpypi.url = "https://test.pypi.org/legacy/"
   ```

5. Publish the repository to TestPyPi, to see that everything works as expected:

   ```console
   poetry publish -r testpypi
   ```

6. Publish the repository to PyPi:

   ```console
   poetry publish -r pypi
   ```
