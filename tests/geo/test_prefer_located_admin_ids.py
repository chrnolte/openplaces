"""Location-derived admin ids win, and an existing id is only a fallback.

A permit export names each record's county in text, and for whole
jurisdictions that text names a neighbor (measured in eastern North
Carolina, 2026-09-16: 94% of one county's permit points lay in the next
county). The point decides wherever it resolves; the text keeps rows
whose point resolves nothing.
"""

import pandas as pd

from openplaces.geo.overlay import prefer_located_admin_ids


def test_location_wins_and_text_fills_only_unlocated_rows():
    index = pd.Index(['r1', 'r2', 'r3', 'r4'], name='record_id')
    located = pd.Series(
        pd.Categorical(['XX-AA-AA', 'XX-AA-BB', None, None]), index=index
    )
    existing = pd.Series(['XX-AA-AA', 'XX-AA-AA', 'XX-AA-BB', None], index=index)

    ids, counts = prefer_located_admin_ids(located, existing)

    assert ids.tolist() == ['XX-AA-AA', 'XX-AA-BB', 'XX-AA-BB', None]
    assert counts == {'located': 2, 'fallback': 1, 'disagree': 1, 'unresolved': 1}
