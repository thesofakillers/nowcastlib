# Nowcast Library - Examples

This directory contains a series of [Jupyter](https://jupyter.org/) notebooks
showcasing the functionality exposed by the Nowcast Library.

## Content

### Rawdata

The notebook [rawdata.ipynb](./rawdata.ipynb) serves to demonstrate the usage of
the [rawdata](../nowcastlib/rawdata.py) submodule in the context of processing
raw data. Particularly, the user is guided through the handling of data incoming
from different data sources, so that it is correctly synchronized and chunked,
while taking into account missing data and different sample rates.

### Triangulation

The notebook [triangulation.ipynb](./triangulation.ipynb) showcases how one may
use the [gis](../nowcastlib/gis.py) and [dynlag](../nowcastlib/dynlag.py)
submodules to simulate data at a target site given data and wind measured at
source site. Because this relies on some knowledge of data synchronization, it
is suggested for readers to first go over the
[rawdata example](./rawdata.ipynb).

### CLI Configurations

Example configuration files for usage with the `nowcastlib` CLI tool are
provided in the format `cli_{command}_config.toml`. For example, to start a
triangulation process from within this `examples/` directory, one may run

```console
nowcastlib triangulate -c ./cli_triangulate_config.yaml
```
