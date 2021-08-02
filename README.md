# Nowcast Library

🧙‍♂️🔧 Utils that can be reused and shared across and beyond the ESO Nowcast
project

This is a public repository hosted on GitHub via a push mirror setup in the
[internal ESO GitLab repository](https://gitlab.eso.org/gstarace/nowcastlib/)

## Installation

Simply run

```console
pip install nowcastlib
```

## Usage and Documentation

Nowcast Library (nowcastlib) consists in a collection of functions organized in
submodules (API) and a tool accessible via the command line (CLI). The latter is
primarily intended for accessing the Nowcast Library Pipeline, an opinionated
yet configurable set of processing steps for wrangling data and evaluating
models in a consistent and rigorous way. More information can be found on the
nowcastlib pipeline index page
([link to markdown](nowcastlib/pipeline/README.md) and
[link to hosted docs](https://giuliostarace.com/nowcastlib/pipeline))

Please refer to the
[examples folder](https://github.com/thesofakillers/nowcastlib/tree/master/examples)
on GitHub for usage examples.

### API

Here is a quick example of how one may import nowcastlib and access to one of
the functions:

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

result = ncl.datasets.compute_trig_fields(data_df, ["A", "C"])
```

More in-depth API documentation can be found
[here](https://giuliostarace.com/nowcastlib/).

### CLI

Some of the library's functionality is bundled in configurable subcommands
accessible via the terminal with the command `nowcastlib`:

```console
usage: nowcastlib [-h] [-v]
                  {triangulate,preprocess,sync,postprocess,datapipe} ...

positional arguments:
  {triangulate,preprocess,sync,postprocess,datapipe}
                        available commands
    triangulate         Run `nowcastlib triangulate -h` for further help
    preprocess          Run `nowcastlib preprocess -h` for further help
    sync                Run `nowcastlib sync -h` for further help
    postprocess         Run `nowcastlib postprocess -h` for further help
    datapipe            Run `nowcastlib datapipe -h` for further help

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase verbosity level from INFO to DEBUG
```

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

To publish changes follow these steps. Ideally this process is automated via a
CI tool triggered by a push/merge to the master branch:

1. Optionally run
   [`poetry version`](https://python-poetry.org/docs/cli/#version) with the
   appropriate argument based on [semver guidelines](https://semver.org/).

2. Update the documentation by running

    ```console
    make document
    ```

3. Prepare the package by running

    ```console
    poetry build
    ```

4. Ensure you have [TestPyPi](https://test.pypi.org/) and PyPi configured as
   your poetry repositories:

    ```console
    poetry config repositories.testpypi https://test.pypi.org/legacy/
    poetry config repositories.pypi https://pypi.org/
    ```

5. Publish the repository to TestPyPi, to see that everything works as expected:

    ```console
    poetry publish -r testpypi
    ```

6. Stage, commit and push your changes (to master) with git.
7. Publish the repository to PyPi:

    ```console
    poetry publish -r pypi
    ```

Upon successful deployment, the library should be available for install via
`pip`
