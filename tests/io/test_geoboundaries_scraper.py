"""Only a permissive geoBoundaries file is ever fetched."""

import pandas as pd

from openplaces.io.scrapers import geoboundaries_scraper as gb


def _table():
    return pd.DataFrame(
        [
            ['KEN', '2', 'permissive', 'open', ''],
            ['FRA', '2', 'share-alike', 'held', 'share-alike'],
            ['XXX', '2', 'unreviewed', 'held', 'not reviewed'],
        ],
        columns=['admin1_id_a3', 'admin_level', 'tier', 'status', 'held_reason'],
    )


def test_only_the_permissive_tier_is_fetched(monkeypatch, tmp_path):
    monkeypatch.setattr(gb, 'load_licenses', _table)
    calls = []
    monkeypatch.setattr(gb.urllib.request, 'urlopen', lambda *a, **k: calls.append(a))
    for iso3 in ('FRA', 'XXX', 'ZZZ'):
        got = gb.fetch(
            target_path=tmp_path / f'{iso3}.gpkg',
            admin_id_to_download='XX',
            admin_level=2,
            iso3=iso3,
        )
        assert got is None
    assert calls == []


def test_the_committed_sidecar_holds_every_share_alike_file():
    table = gb.load_licenses()
    assert set(table['tier']) <= {'permissive', 'share-alike', 'unreviewed'}
    assert (table.loc[table['tier'] != 'permissive', 'status'] == 'held').all()
    assert (table.loc[table['tier'] == 'permissive', 'status'] == 'open').all()
    assert (
        not table.loc[table['tier'] == 'permissive', 'license_spdx']
        .str.contains('SA|ODbL', regex=True)
        .any()
    )
