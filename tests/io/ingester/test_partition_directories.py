"""Which admin unit the external and heap directories are keyed on.

The two had drifted apart: the external directory moved to the partition's own
download unit while the heap directory stayed on the recipe's, so a recipe with
a fixed `uncompressed_file_name` resolved to one heap path shared by every
state and a leftover extraction satisfied the next state's "already unzipped"
skip. Keying both on the download unit closes that; a
`partition_key_transformation`, which deliberately points several download
units at one file, keeps both at the recipe's own unit so the shared download
is fetched once.
"""

from openplaces.core.schema import AdminId, Entity
from openplaces.io import ingester as ingester_module
from openplaces.io.ingester import Ingester
from openplaces.timing import Timer


def _ingester(recipe, admin_id_to_download, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingester_module,
        'heap_dir',
        lambda admin_id, *a, **k: tmp_path / 'heap' / str(admin_id),
    )
    monkeypatch.setattr(
        ingester_module,
        'external_dir',
        lambda admin_id, *a, **k: tmp_path / 'external' / str(admin_id),
    )
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = recipe
    ingester.timer = Timer('test')
    ingester.verbose = False
    ingester._owns_timer = False
    ingester.download_partition = {
        'admin_id_to_download': admin_id_to_download,
        'partition_id_to_download': None,
        'download_url': None,
    }
    return ingester


def _recipe(**download_by):
    return {
        'admin_id': AdminId('US'),
        'entity': Entity('footprint'),
        'download_by': download_by,
        'uncompressed_file_name': 'gis_osm_buildings_a_free_1.shp',
    }


def test_two_states_do_not_share_one_heap_path(tmp_path, monkeypatch):
    recipe = _recipe(admin_level=2)
    first = _ingester(recipe, 'US-NC', tmp_path, monkeypatch)
    first._resolve_downloaded_and_data_paths()
    second = _ingester(recipe, 'US-TX', tmp_path, monkeypatch)
    second._resolve_downloaded_and_data_paths()

    assert first.recipe_heap_dir != second.recipe_heap_dir
    assert (
        first.recipe_heap_dir
        == first.recipe_external_dir.parents[1] / ('heap') / 'US-NC'
    )


def test_a_shared_download_keeps_one_directory(tmp_path, monkeypatch):
    recipe = _recipe(
        admin_level=3,
        partition_key_transformation={
            'admin3_id_admin1': {'operation': 'substring', 'args': [0, 5]}
        },
    )
    first = _ingester(recipe, 'US-MA-MI-AA', tmp_path, monkeypatch)
    first._resolve_downloaded_and_data_paths()
    second = _ingester(recipe, 'US-MA-MI-BB', tmp_path, monkeypatch)
    second._resolve_downloaded_and_data_paths()

    assert first.recipe_external_dir == second.recipe_external_dir
    assert first.recipe_external_dir == tmp_path / 'external' / 'US'


def test_the_two_directories_agree_on_the_unit(tmp_path, monkeypatch):
    ingester = _ingester(_recipe(admin_level=2), 'US-NC', tmp_path, monkeypatch)
    ingester._resolve_downloaded_and_data_paths()

    assert ingester.recipe_heap_dir.name == ingester.recipe_external_dir.name
