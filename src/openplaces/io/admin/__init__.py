"""
Administrative referencing and mapping: the global admin spine, the
ids that name its units, and the layers that draw them.

Until 2026-09-22 this was one 2,087-line module. Its sections are now
ids (index builders per level), names (cleaning), generate (the id
waterfall), spine (updating the committed spine) and context (map
layers); wikidata and units moved in from beside it. The package
file re-exports every public name the module exported, so
``from openplaces.io.admin import <name>`` still works. A test that
monkeypatches a name a function looks up patches the submodule the
function lives in, not this file.
"""

from openplaces.io.admin.context import context_layers as context_layers
from openplaces.io.admin.generate import generate_admin_ids as generate_admin_ids
from openplaces.io.admin.ids import (
    admin1_id_index_from_admin1_id_a3 as admin1_id_index_from_admin1_id_a3,
)
from openplaces.io.admin.ids import (
    admin2_id_index_from_admin2_gadm as admin2_id_index_from_admin2_gadm,
)
from openplaces.io.admin.ids import (
    admin3_id_index_from_admin3_code as admin3_id_index_from_admin3_code,
)
from openplaces.io.admin.ids import (
    admin3_id_index_from_admin3_gadm as admin3_id_index_from_admin3_gadm,
)
from openplaces.io.admin.ids import (
    admin3_id_index_from_local as admin3_id_index_from_local,
)
from openplaces.io.admin.ids import (
    admin4_id_index_from_gb_ons as admin4_id_index_from_gb_ons,
)
from openplaces.io.admin.ids import (
    admin4_id_index_from_gisco_lau as admin4_id_index_from_gisco_lau,
)
from openplaces.io.admin.ids import get_admin1_iso as get_admin1_iso
from openplaces.io.admin.ids import get_admin2_iso as get_admin2_iso
from openplaces.io.admin.ids import (
    nuts_crosswalk_index_from_admin4 as nuts_crosswalk_index_from_admin4,
)
from openplaces.io.admin.names import clean_geographic_name as clean_geographic_name
from openplaces.io.admin.names import fold_to_ascii as fold_to_ascii
from openplaces.io.admin.spine import update_admin_spine as update_admin_spine
