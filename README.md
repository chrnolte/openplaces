# openplaces

[![PyPI version](https://img.shields.io/pypi/v/openplaces.svg)](https://pypi.org/project/openplaces/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://openplaces.readthedocs.io/en/latest/)

**openplaces** is an open-source data and analytics platform for integrating property data (parcels, buildings, transactions), environmental indicators, and socio‑economic data at scale.

Its principal purpose is to support reproducible research for conservation, land policy, and environmental analytics, anywhere.

**openplaces** is maintained by researchers at Boston University and released under the **Apache-2.0** license.

## Features

- **Recipe-driven pipeline**: YAML recipes declare how each dataset is ingested, harmonized,
  enriched, and curated, so new sources are added without writing pipeline code.
- **Entity coverage**: parcels, buildings, footprints, dwellings, properties, and transactions,
  plus environmental and socio-economic datasets (terrain, land cover, water, climate, risk,
  population).
- **Evidence-based curation**: reconciles competing sources, imputes gaps, and infers canonical
  attributes — including occupancy, story count, and roof shape from imagery-based models —
  while keeping every source's raw evidence traceable.
- **Globally consistent, hierarchical administrative referencing** (e.g. `US-MA-MI-CA`) that
  ties sources across geographies together and supports international comparative research.
- **Cluster-friendly workflows**: the same recipe runs unchanged on a laptop or scales out to a
  research cluster.
- **Recipe catalog** spanning dozens of U.S. states and counties, plus Colombia, Germany,
  France, and the United Kingdom, contributed by the research community.
- **Modern Python stack** (`geopandas`, `duckdb`, `rasterio`, `pyarrow`, `pyproj`).

## Quick start

```python
import openplaces as op

# Ingest, then read back an entity for one administrative unit
op.ingest('US-MA_parcel-massgis-2025', admin_ids=['US-MA-MI'])
parcels = op.get_entities('US-MA_parcel-massgis-2025', admin_id='US-MA-MI')
```

Recipes are looked up by ID (`{admin_id}_{entity}-{source}-{version}`); browse the full
catalog in the docs linked below.

## Documentation

- User guide: https://docs.openplaces.io

## Installation

- Get started: https://docs.openplaces.io/en/latest/2_get-started

## Governance

`openplaces` is developed by a network of contributors, including researchers
affiliated with Boston University. Ownership of the intellectual property in this
repository — including any Boston University interest arising from grant-funded
or in-scope faculty research — is under institutional review; see
[DISCLAIMER.md](DISCLAIMER.md). Nothing on this page should be read as a final
determination of ownership by any party.

Contact **contact@openplaces.io** if you would like to get involved.

## License

Released under the **Apache License 2.0**. See [LICENSE.md](LICENSE.md) for details.

See [DISCLAIMER.md](DISCLAIMER.md) for additional disclaimers, including on data
privacy, IP ownership, and the project's intended scope of use.

## Citation

If you use **openplaces** in academic work, please cite:

```bibtex
@misc{openplaces2026,
  author       = {Christoph Nolte et al.},
  title        = {openplaces: an open-source processing engine for land, building, and property data},
  year         = {2026},
  howpublished = {\url{https://github.com/chrnolte/openplaces}}
}
```

## Acknowledgments

This codebase combines code contributions developed under four federal research grants, supported by the U.S. National Science Foundation (NSF) and the National Aeronautics and Space Administration (NASA).

For more information, visit: https://placeslab.org/research

Any opinions, findings, and conclusions or recommendations expressed are those of the authors and do not necessarily reflect the views of the supporting agencies.

## A note on AI tools

The human authors of `openplaces` have relied on AI tools throughout its development: Claude as a coding and thinking partner, and Gemini and Codex to clean up comments and docs. We wouldn't be as far along without them, and we'd rather say so plainly than leave it unstated. The authors reviewed the design and results and are responsible for them.
