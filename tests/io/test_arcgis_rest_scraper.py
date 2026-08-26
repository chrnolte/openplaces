"""Tests for the generic ArcGIS REST layer scraper.

Focus on the streaming writer and the completeness guard: a flaky
service previously yielded a small fraction of a layer while reporting
success, because every transient failure was treated as a permanent bad
record.
"""

import json

import pandas as pd
import pytest

from openplaces.io.scrapers import arcgis_rest_scraper as scraper


def _feature(i):
    return {
        'type': 'Feature',
        'properties': {'OBJECTID': i},
        'geometry': {'type': 'Point', 'coordinates': [float(i), 0.0]},
    }


@pytest.fixture
def fake_service(monkeypatch):
    """Stub `_get_json` with a controllable in-memory layer."""

    state = {
        'total': 0,
        'max_record_count': 2,
        'fail_offsets': set(),
        'calls': [],
        'retries_seen': [],
        'attribute_queries': [],
    }

    def _get_json(url, params, *, timeout, retries, verbose, label):
        if not url.endswith('/query'):
            return {'maxRecordCount': state['max_record_count']}
        if params.get('returnCountOnly'):
            return {'count': state['total']}

        offset = int(params['resultOffset'])
        count = int(params['resultRecordCount'])
        state['calls'].append((offset, count))
        state['retries_seen'].append(retries)

        if offset in state['fail_offsets']:
            raise RuntimeError('permanent failure')

        end = min(offset + count, state['total'])
        if params.get('returnGeometry') == 'false':
            # Attribute-only join query.
            state['attribute_queries'].append(params.get('outFields'))
            return {
                'attributes_are': 'flat',
                'features': [
                    {'attributes': {'PIN': f'p{i}', 'EXTRA': i * 10}}
                    for i in range(offset, end)
                ],
            }
        return {'features': [_feature(i) for i in range(offset, end)]}

    monkeypatch.setattr(scraper, '_get_json', _get_json)
    return state


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_streams_all_pages_in_order(fake_service, tmp_path):
    fake_service['total'] = 5
    out = tmp_path / 'layer.geojson'

    scraper.fetch(target_path=out, layer_url='http://svc/0')

    doc = _read(out)
    assert doc['type'] == 'FeatureCollection'
    assert [f['properties']['OBJECTID'] for f in doc['features']] == list(range(5))


def test_zero_features_writes_valid_empty_collection(fake_service, tmp_path):
    fake_service['total'] = 0
    out = tmp_path / 'layer.geojson'

    scraper.fetch(target_path=out, layer_url='http://svc/0')

    assert _read(out) == {'type': 'FeatureCollection', 'features': []}


def test_page_size_override_is_honored(fake_service, tmp_path):
    fake_service['total'] = 6
    fake_service['max_record_count'] = 2
    out = tmp_path / 'layer.geojson'

    scraper.fetch(target_path=out, layer_url='http://svc/0', page_size=3)

    assert [c[1] for c in fake_service['calls'] if c[1] == 3]
    assert len(_read(out)['features']) == 6


def test_page_fetches_get_the_full_retry_budget(fake_service, tmp_path):
    """Each range must retry before bisecting.

    The regression this pins: `_fetch_range` hardcoded `retries=1`, so a
    transient failure on a flaky service was treated as a permanent bad
    record and bisected away to skips instead of being retried. The
    retry loop lives inside `_get_json`, so the observable contract is
    the budget handed to it.
    """
    fake_service['total'] = 4
    fake_service['max_record_count'] = 2
    out = tmp_path / 'layer.geojson'

    scraper.fetch(target_path=out, layer_url='http://svc/0', retries=5)

    assert fake_service['retries_seen']
    assert set(fake_service['retries_seen']) == {5}


def test_incomplete_download_raises_and_leaves_no_file(fake_service, tmp_path):
    fake_service['total'] = 4
    fake_service['max_record_count'] = 1
    fake_service['fail_offsets'] = {1, 2}
    out = tmp_path / 'layer.geojson'

    with pytest.raises(RuntimeError, match='incomplete download'):
        scraper.fetch(target_path=out, layer_url='http://svc/0')

    assert not out.exists()
    assert not out.with_name(out.name + '.part').exists()


def test_allow_partial_saves_the_short_layer(fake_service, tmp_path):
    fake_service['total'] = 4
    fake_service['max_record_count'] = 1
    fake_service['fail_offsets'] = {1, 2}
    out = tmp_path / 'layer.geojson'

    scraper.fetch(target_path=out, layer_url='http://svc/0', allow_partial=True)

    ids = [f['properties']['OBJECTID'] for f in _read(out)['features']]
    assert ids == [0, 3]


def test_empty_page_between_pages_keeps_json_valid(fake_service, tmp_path):
    """A bisected range returning nothing must not emit a stray comma."""
    fake_service['total'] = 3
    fake_service['max_record_count'] = 1
    fake_service['fail_offsets'] = {0}
    out = tmp_path / 'layer.geojson'

    scraper.fetch(target_path=out, layer_url='http://svc/0', allow_partial=True)

    ids = [f['properties']['OBJECTID'] for f in _read(out)['features']]
    assert ids == [1, 2]


# Bulk download and attribute join


@pytest.fixture
def bulk_geojson(monkeypatch):
    """Stub `_download_bulk` with a small on-disk FeatureCollection."""
    written = {}

    def _download_bulk(url, target, *, timeout, retries, verbose, label):
        written['url'] = url
        target.write_text(
            json.dumps(
                {
                    'type': 'FeatureCollection',
                    'features': [
                        {
                            'type': 'Feature',
                            'properties': {'PIN': f'p{i}'},
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [float(i), 0.0],
                            },
                        }
                        for i in range(4)
                    ],
                }
            ),
            encoding='utf-8',
        )

    monkeypatch.setattr(scraper, '_download_bulk', _download_bulk)
    return written


def test_bulk_url_skips_paging_entirely(bulk_geojson, fake_service, tmp_path):
    out = tmp_path / 'layer.geojson'

    scraper.fetch(target_path=out, layer_url='http://svc/0', bulk_url='http://svc/bulk')

    assert bulk_geojson['url'] == 'http://svc/bulk'
    assert fake_service['calls'] == []
    assert len(_read(out)['features']) == 4


def test_attribute_join_adds_columns_without_geometry(
    bulk_geojson, fake_service, tmp_path
):
    fake_service['total'] = 4
    out = tmp_path / 'layer.geojson'

    scraper.fetch(
        target_path=out,
        layer_url='http://svc/0',
        bulk_url='http://svc/bulk',
        attribute_join={'key': 'PIN', 'fields': ['EXTRA']},
    )

    doc = _read(out)
    joined = {f['properties']['PIN']: f['properties']['EXTRA'] for f in doc['features']}
    assert joined == {'p0': 0, 'p1': 10, 'p2': 20, 'p3': 30}
    # The join query must never ask for geometry -- that is the whole point.
    assert fake_service['attribute_queries'] == ['PIN,EXTRA']


def test_attribute_join_below_min_match_raises(bulk_geojson, fake_service, tmp_path):
    """A wrong key yields null columns; that must fail, not pass quietly."""
    fake_service['total'] = 1  # only p0 comes back, 25% of 4 features
    out = tmp_path / 'layer.geojson'

    with pytest.raises(RuntimeError, match='matched only'):
        scraper.fetch(
            target_path=out,
            layer_url='http://svc/0',
            bulk_url='http://svc/bulk',
            attribute_join={'key': 'PIN', 'fields': ['EXTRA']},
        )


def test_attribute_join_rejects_colliding_field(bulk_geojson, fake_service, tmp_path):
    fake_service['total'] = 4
    out = tmp_path / 'layer.geojson'

    with pytest.raises(ValueError, match='already exist'):
        scraper.fetch(
            target_path=out,
            layer_url='http://svc/0',
            bulk_url='http://svc/bulk',
            attribute_join={'key': 'PIN', 'fields': ['PIN']},
        )


def test_attribute_join_requires_key_and_fields(bulk_geojson, fake_service, tmp_path):
    out = tmp_path / 'layer.geojson'

    with pytest.raises(ValueError, match='attribute_join needs'):
        scraper.fetch(
            target_path=out,
            layer_url='http://svc/0',
            bulk_url='http://svc/bulk',
            attribute_join={'fields': ['EXTRA']},
        )


def test_count_request_forces_json_format(monkeypatch, tmp_path):
    """The count query must not inherit the feature pages' `f=geojson`.

    Most ArcGIS servers ignore `returnCountOnly` in GeoJSON mode and answer
    with an ordinary FeatureCollection, so reading `count` off the response
    raised KeyError and failed the whole download. The shared `fake_service`
    fixture answers any count request regardless of `f`, which is more
    forgiving than a real service and cannot catch this.
    """
    seen_formats = []

    def _get_json(url, params, *, timeout, retries, verbose, label):
        if not url.endswith('/query'):
            return {'maxRecordCount': 2}
        if params.get('returnCountOnly'):
            seen_formats.append(params.get('f'))
            if params.get('f') != 'json':
                # What a real server sends back in GeoJSON mode.
                return {'type': 'FeatureCollection', 'properties': {}, 'features': []}
            return {'count': 1}
        offset = int(params['resultOffset'])
        end = min(offset + int(params['resultRecordCount']), 1)
        return {
            'type': 'FeatureCollection',
            'features': [_feature(i) for i in range(offset, end)],
        }

    monkeypatch.setattr(scraper, '_get_json', _get_json)
    out = tmp_path / 'layer.geojson'

    scraper.fetch(target_path=out, layer_url='http://svc/0')

    assert seen_formats == ['json']
    assert len(_read(out)['features']) == 1


def test_resolve_layer_url_substitutes_admin_key(monkeypatch):
    """`{admin_key}` in `layer_url` resolves via `get_admin`.

    Covers the case a shared-service `admin_id_column` filter cannot: a
    statewide source published as one genuinely separate FeatureServer
    per admin unit (e.g. Utah's per-county LIR parcel services), named
    after the admin unit.
    """
    seen = {}

    def fake_get_admin(admin_id, level, columns=None):
        seen['admin_id'] = str(admin_id)
        seen['level'] = level
        seen['columns'] = columns
        return pd.DataFrame({columns: ['Salt Lake']})

    monkeypatch.setattr(scraper, 'get_admin', fake_get_admin)

    resolved = scraper._resolve_layer_url(
        'https://svc/Parcels_{admin_key}_LIR/FeatureServer/0',
        admin_id_to_download='US-UT-SL',
        admin_key_column='name',
        admin_key_transform='remove_spaces',
    )

    assert resolved == 'https://svc/Parcels_SaltLake_LIR/FeatureServer/0'
    assert seen == {'admin_id': 'US-UT-SL', 'level': 3, 'columns': 'name'}


def test_resolve_layer_url_without_placeholder_is_a_no_op():
    url = 'https://svc/Parcels/FeatureServer/0'
    assert (
        scraper._resolve_layer_url(
            url,
            admin_id_to_download=None,
            admin_key_column=None,
            admin_key_transform=None,
        )
        == url
    )


def test_resolve_layer_url_requires_admin_key_column():
    with pytest.raises(ValueError, match='admin_key_column'):
        scraper._resolve_layer_url(
            'https://svc/Parcels_{admin_key}_LIR/FeatureServer/0',
            admin_id_to_download='US-UT-SL',
            admin_key_column=None,
            admin_key_transform=None,
        )


def test_resolve_layer_url_rejects_unknown_transform(monkeypatch):
    monkeypatch.setattr(
        scraper, 'get_admin', lambda *a, **k: pd.DataFrame({'name': ['Salt Lake']})
    )
    with pytest.raises(NotImplementedError, match='uppercase'):
        scraper._resolve_layer_url(
            'https://svc/Parcels_{admin_key}_LIR/FeatureServer/0',
            admin_id_to_download='US-UT-SL',
            admin_key_column='name',
            admin_key_transform='uppercase',
        )


def test_fetch_resolves_admin_key_layer_url(fake_service, tmp_path, monkeypatch):
    """`fetch()` itself wires `admin_key_column`/`admin_key_transform`
    through to the layer it pages, not just `_resolve_layer_url` alone."""
    fake_service['total'] = 2
    seen_urls = []
    real_get_json = scraper._get_json

    def spying_get_json(url, params, **kwargs):
        seen_urls.append(url)
        return real_get_json(url, params, **kwargs)

    monkeypatch.setattr(scraper, '_get_json', spying_get_json)
    monkeypatch.setattr(
        scraper, 'get_admin', lambda *a, **k: pd.DataFrame({'name': ['Salt Lake']})
    )
    out = tmp_path / 'layer.geojson'

    scraper.fetch(
        target_path=out,
        layer_url='http://svc/Parcels_{admin_key}_LIR/FeatureServer/0',
        admin_id_to_download='US-UT-SL',
        admin_key_column='name',
        admin_key_transform='remove_spaces',
    )

    assert all('Parcels_SaltLake_LIR' in url for url in seen_urls)


def test_resolve_where_builds_admin_clause(monkeypatch):
    """`where_admin_column` filters a *shared* service by an in-data

    admin column -- the complementary case to `{admin_key}`, which is
    for N genuinely different services (e.g. Arkansas's single CAMP
    layer for all 75 counties, filtered by its own `countyfips`).
    """
    monkeypatch.setattr(
        scraper,
        'get_admin',
        lambda *a, **k: pd.DataFrame({'admin3_id_admin1': ['05101']}),
    )

    resolved = scraper._resolve_where(
        '1=1',
        admin_id_to_download='US-AR-NE',
        admin_key_column='admin3_id_admin1',
        admin_key_transform=None,
        where_admin_column='countyfips',
    )

    assert resolved == "countyfips = '05101'"


def test_resolve_where_ands_onto_an_existing_clause(monkeypatch):
    monkeypatch.setattr(
        scraper,
        'get_admin',
        lambda *a, **k: pd.DataFrame({'admin3_id_admin1': ['05101']}),
    )

    resolved = scraper._resolve_where(
        'LOT_TYPE = 2',
        admin_id_to_download='US-AR-NE',
        admin_key_column='admin3_id_admin1',
        admin_key_transform=None,
        where_admin_column='countyfips',
    )

    assert resolved == "(LOT_TYPE = 2) AND (countyfips = '05101')"


def test_resolve_where_without_column_is_a_no_op():
    assert (
        scraper._resolve_where(
            '1=1',
            admin_id_to_download=None,
            admin_key_column=None,
            admin_key_transform=None,
            where_admin_column=None,
        )
        == '1=1'
    )


def test_resolve_where_requires_admin_key_column():
    with pytest.raises(ValueError, match='admin_key_column'):
        scraper._resolve_where(
            '1=1',
            admin_id_to_download='US-AR-NE',
            admin_key_column=None,
            admin_key_transform=None,
            where_admin_column='countyfips',
        )


def test_fetch_resolves_where_admin_column(fake_service, tmp_path, monkeypatch):
    """`fetch()` itself wires `where_admin_column` into the query params,

    not just `_resolve_where` alone."""
    fake_service['total'] = 2
    seen_params = []
    real_get_json = scraper._get_json

    def spying_get_json(url, params, **kwargs):
        seen_params.append(params)
        return real_get_json(url, params, **kwargs)

    monkeypatch.setattr(scraper, '_get_json', spying_get_json)
    monkeypatch.setattr(
        scraper,
        'get_admin',
        lambda *a, **k: pd.DataFrame({'admin3_id_admin1': ['05101']}),
    )
    out = tmp_path / 'layer.geojson'

    scraper.fetch(
        target_path=out,
        layer_url='http://svc/Planning_Cadastre/FeatureServer/6',
        admin_id_to_download='US-AR-NE',
        admin_key_column='admin3_id_admin1',
        where_admin_column='countyfips',
    )

    wheres = [p['where'] for p in seen_params if 'where' in p]
    assert wheres
    assert all(w == "countyfips = '05101'" for w in wheres)


# attribute_join: comparing on a normalized key


def _write_points(path, pins):
    """Write a tiny point FeatureCollection keyed on PIN."""
    path.write_text(
        json.dumps(
            {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'properties': {'PIN': pin},
                        'geometry': {'type': 'Point', 'coordinates': [i, 0.0]},
                    }
                    for i, pin in enumerate(pins)
                ],
            }
        ),
        encoding='utf-8',
    )


def test_key_conv_matches_across_a_separator_difference(tmp_path):
    """The Maine case: one submission punctuates a map-lot with dashes and
    the other with underscores, so a literal comparison matches nothing."""
    out = tmp_path / 'layer.geojson'
    _write_points(out, ['012-345', '012-346', '012-347'])
    table = {f'012_{n}': {'EXTRA': int(n)} for n in ('345', '346', '347')}

    scraper._apply_attribute_join(
        out,
        table,
        key='PIN',
        fields=['EXTRA'],
        min_match=0.9,
        verbose=False,
        label='t',
        key_conv='pipe',
    )

    doc = _read(out)
    joined = {f['properties']['PIN']: f['properties']['EXTRA'] for f in doc['features']}
    assert joined == {'012-345': 345, '012-346': 346, '012-347': 347}


def test_key_conv_leaves_the_sources_own_key_column_intact(tmp_path):
    out = tmp_path / 'layer.geojson'
    _write_points(out, ['012-345'])
    scraper._apply_attribute_join(
        out,
        {'012_345': {'EXTRA': 1}},
        key='PIN',
        fields=['EXTRA'],
        min_match=0.5,
        verbose=False,
        label='t',
        key_conv='pipe',
    )
    props = _read(out)['features'][0]['properties']
    assert props['PIN'] == '012-345'
    assert scraper._JOIN_KEY_COLUMN not in props


def test_key_conv_drops_attribute_rows_that_become_ambiguous(tmp_path):
    """Normalizing makes keys collide that did not collide before. Two
    assessor rows reaching one key would each attach to every parcel
    carrying it, so they are dropped rather than picked between."""
    out = tmp_path / 'layer.geojson'
    _write_points(out, ['012-345', '012-999'])
    table = {
        '012-345': {'EXTRA': 1},
        '012_345': {'EXTRA': 2},  # collides with the row above under pipe
        '012_999': {'EXTRA': 9},
    }

    scraper._apply_attribute_join(
        out,
        table,
        key='PIN',
        fields=['EXTRA'],
        min_match=0.4,
        verbose=False,
        label='t',
        key_conv='pipe',
    )

    doc = _read(out)
    assert len(doc['features']) == 2, 'an ambiguous key must not multiply rows'
    joined = {f['properties']['PIN']: f['properties']['EXTRA'] for f in doc['features']}
    assert joined['012-999'] == 9
    assert joined['012-345'] is None


def test_key_conv_that_loses_matches_raises(tmp_path):
    """A conversion is only ever meant to recover matches. One that costs
    them is the wrong conversion, and saying so beats a quiet regression."""
    out = tmp_path / 'layer.geojson'
    _write_points(out, ['12-3', '1-23'])
    # Both sides agree literally; 'simple' collapses them onto one key,
    # which then reads as ambiguous and is dropped.
    table = {'12-3': {'EXTRA': 1}, '1-23': {'EXTRA': 2}}

    with pytest.raises(RuntimeError, match='is the wrong conversion'):
        scraper._apply_attribute_join(
            out,
            table,
            key='PIN',
            fields=['EXTRA'],
            min_match=0.0,
            verbose=False,
            label='t',
            key_conv='simple',
        )


def test_no_key_conv_is_unchanged_behaviour(tmp_path):
    out = tmp_path / 'layer.geojson'
    _write_points(out, ['012-345'])
    scraper._apply_attribute_join(
        out,
        {'012-345': {'EXTRA': 7}},
        key='PIN',
        fields=['EXTRA'],
        min_match=0.9,
        verbose=False,
        label='t',
    )
    props = _read(out)['features'][0]['properties']
    assert props == {'PIN': '012-345', 'EXTRA': 7}


# attribute_join: a key rebuilt from several fields


def _write_features(path, rows):
    """Write a point FeatureCollection whose properties are `rows`."""
    path.write_text(
        json.dumps(
            {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'properties': props,
                        'geometry': {'type': 'Point', 'coordinates': [i, 0.0]},
                    }
                    for i, props in enumerate(rows)
                ],
            }
        ),
        encoding='utf-8',
    )


def test_composite_key_joins_on_the_parts(tmp_path):
    """The Acton case: both sides agree on the town code and on the
    map-lot, and disagree on the single field that concatenates them, so
    rebuilding the key from its parts is what reaches the rows."""
    out = tmp_path / 'layer.geojson'
    _write_features(
        out,
        [
            {'GEOCODE': '31010', 'LOT': '001-002', 'STATE_ID': 'junk-1'},
            {'GEOCODE': '31010', 'LOT': '001-003', 'STATE_ID': 'junk-2'},
        ],
    )
    table = {
        f'31010{scraper._KEY_PART_SEPARATOR}001-002': {'EXTRA': 1},
        f'31010{scraper._KEY_PART_SEPARATOR}001-003': {'EXTRA': 2},
    }

    scraper._apply_attribute_join(
        out,
        table,
        key=['GEOCODE', 'LOT'],
        fields=['EXTRA'],
        min_match=0.9,
        verbose=False,
        label='t',
    )

    joined = {
        f['properties']['LOT']: f['properties']['EXTRA'] for f in _read(out)['features']
    }
    assert joined == {'001-002': 1, '001-003': 2}


def test_composite_key_parts_cannot_bleed_into_each_other(tmp_path):
    """('1', '2-3') and ('1-2', '3') must stay distinct -- joining the
    parts on a hyphen would collapse them onto one key."""
    out = tmp_path / 'layer.geojson'
    _write_features(out, [{'A': '1', 'B': '2-3'}, {'A': '1-2', 'B': '3'}])
    table = {f'1{scraper._KEY_PART_SEPARATOR}2-3': {'EXTRA': 1}}

    scraper._apply_attribute_join(
        out,
        table,
        key=['A', 'B'],
        fields=['EXTRA'],
        min_match=0.4,
        verbose=False,
        label='t',
    )

    joined = {
        f['properties']['A']: f['properties']['EXTRA'] for f in _read(out)['features']
    }
    assert joined == {'1': 1, '1-2': None}


def test_composite_key_combines_with_key_conv(tmp_path):
    out = tmp_path / 'layer.geojson'
    _write_features(out, [{'GEOCODE': '31010', 'LOT': '001-002'}])
    table = {f'31010{scraper._KEY_PART_SEPARATOR}001_002': {'EXTRA': 5}}

    scraper._apply_attribute_join(
        out,
        table,
        key=['GEOCODE', 'LOT'],
        fields=['EXTRA'],
        min_match=0.9,
        verbose=False,
        label='t',
        key_conv='pipe',
    )

    props = _read(out)['features'][0]['properties']
    assert props['EXTRA'] == 5
    assert props['GEOCODE'] == '31010' and props['LOT'] == '001-002'


def test_composite_key_names_a_missing_field(tmp_path):
    out = tmp_path / 'layer.geojson'
    _write_features(out, [{'GEOCODE': '31010'}])
    with pytest.raises(ValueError, match=r"\['LOT'\]"):
        scraper._apply_attribute_join(
            out,
            {},
            key=['GEOCODE', 'LOT'],
            fields=['EXTRA'],
            min_match=0.0,
            verbose=False,
            label='t',
        )


def test_composite_key_is_requested_from_the_service(fake_service):
    """Every part of the key has to be in `outFields`, or the attribute
    side cannot assemble the same string the geometry side does."""
    fake_service['total'] = 4
    scraper._fetch_attribute_table(
        'http://svc/9',
        key=['PIN', 'EXTRA'],
        fields=['EXTRA'],
        where='1=1',
        page_size=2,
        timeout=1,
        retries=1,
        verbose=False,
        label='t',
    )
    assert fake_service['attribute_queries'] == ['PIN,EXTRA,EXTRA'] * 2
