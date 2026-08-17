"""Tests for the generic ArcGIS REST layer scraper.

Focus on the streaming writer and the completeness guard: a flaky
service previously yielded a small fraction of a layer while reporting
success, because every transient failure was treated as a permanent bad
record.
"""

import json

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
