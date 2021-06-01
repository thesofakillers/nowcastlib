# Nowcast Library - Examples

This directory contains a series of [Jupyter](https://jupyter.org/) notebooks
and config files showcasing the functionality exposed by the Nowcast Library.

To get interactive plots when running the Jupyter notebooks, please change the
first line from

```python
%matplotlib inline
```

to

```python
%matplotlib widget
```

## Content

### Data Synchronization

The notebook [datasync.ipynb](./datasync.ipynb) serves to demonstrate the usage
of the [datasets](../nowcastlib/datasets.py) submodule in the context of syncing
datasets. Particularly, the user is guided through the handling of data incoming
from different data sources, so that it is correctly synchronized and chunked,
while taking into account missing data and different sample rates.

### Triangulation

The notebook [triangulation.ipynb](./triangulation.ipynb) showcases how one may
use the [gis](../nowcastlib/gis.py) and [dynlag](../nowcastlib/dynlag.py)
submodules to simulate data at a target site given data and wind measured at
source site. Because this relies on some knowledge of data synchronization, it
is suggested for readers to first go over the
[data sync example](./datasync.ipynb).

### Signals

The notebook [signals.ipynb](./signals.ipynb) demonstrates how one may use the
[signals](../nowcastlib/signals.py) submodule to add a mixture of noise to an
underlying signal while achieving a requested signal-to-noise ratio. Readers are
encouraged to experiment with the `gen_composite_red_noise()` function.

### CLI Configurations

Example configuration files for usage with the `nowcastlib` CLI tool are
provided in the format `cli_{command}_config.toml`. For example, from within
this `examples/` directory, one may run

```console
nowcastlib triangulate -c ./cli_triangulate_config.yaml
```

to start a triangulation process.
