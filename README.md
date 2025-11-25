# openplaces

[![PyPI version](https://img.shields.io/pypi/v/openplaces.svg)](https://pypi.org/project/openplaces/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://openplaces.readthedocs.io/en/latest/)

**openplaces** is an open-source data and analytics platform for integrating parcel boundaries, environmental indicators, and socio‑economic data at scale.

Maintained by researchers at academic institutions and released under the **Apache-2.0** license, it supports reproducible research for conservation, land policy, and environmental analytics.

This repository is an inital commit with a sceleton structure and will be populated with code from sister projects 2025-2026.

---

## Features (goals)

- Standardized **parcel-level** data ingestion and harmonization.
- Connectors to **remote-sensing archives** (Landsat (LS), Sentinel, NAIP) and environmental datasets.
- Utilities for **valuation modeling**, **conservation planning**, and spatial statistics.
- Cloud/cluster-friendly workflows for **reproducibility** (containers, CI, distributed compute).
- Modern Python stack (GeoPandas, Rasterio, Xarray, Dask, Scikit‑learn/PyTorch).

---

## Documentation

- User guide: https://openplaces.readthedocs.io/

---

## Installation

From source (development):

```bash
git clone https://github.com/chrnolte/openplaces.git
cd openplaces
python dev.py setup
```

From PyPI (future):

```bash
pip install openplaces
```

Optional (recommended) extras:

```bash
# Example extras; adjust to your setup
pip install "openplaces[dev,docs]"
```

---

## Quick start

> The snippet below is illustrative for intended use — consult the docs for the current interface.

```python
import openplaces.api as op

# Load parcels for Middlesex, Massachusetts, United States
parcels = op.load_parcels("US-MA-MI")

# Join satellite-derived forest change
parcels = op.join_forest_change(parcels)

# Get sales dataset for hedonic analysis
sales = op.get_sales(parcels)

# Estimate land values
values = op.estimate_land_values(parcels)
print(values.describe())
```

---

## Governance & Sustainability

Currently developed by an informal network of academic collaborators; new partners are welcome.

Contact: **contact@openplaces.io**

---

## License

Released under the **Apache License 2.0**. See [LICENSE.md](LICENSE.md) for details.  
© 2025 Christoph Nolte

---

## Citation

If you use **openplaces** in academic work, please cite:

```bibtex
@misc{openplaces2025,
  author       = {Christoph Nolte},
  title        = {openplaces: Global property data and analytics platform},
  year         = {2025},
  howpublished = {\url{https://github.com/chrnolte/openplaces}}
}
```

---

## Acknowledgments

This codebases brings together code contributions created under various research grants supported by the U.S. National Science Foundation (NSF) and the National Aeronautics and Space Administration (NASA), together with partner institutions across multiple countries.

Any opinions, findings, and conclusions or recommendations expressed are those of the authors and do not necessarily reflect the views of the supporting agencies.
