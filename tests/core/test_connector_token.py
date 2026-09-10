"""A connector's provenance token: `{connector}:{model}`, qualified by the
same `+` grammar as every other route token."""

import pytest

from openplaces.core.provenance import connector_token, is_imputed, mark_imputed


def test_token_shape():
    assert (
        connector_token('openplaces-valuation', 'et_v2') == 'openplaces-valuation:et_v2'
    )
    assert connector_token('openplaces-climate-risk') == 'openplaces-climate-risk'


def test_the_plus_grammar_still_applies():
    token = mark_imputed(connector_token('openplaces-valuation', 'et_v2'))
    assert token == 'openplaces-valuation:et_v2+imputed'
    assert is_imputed([token, 'openplaces-valuation:et_v2']).tolist() == [True, False]


@pytest.mark.parametrize('bad', ['a+b', 'a:b'])
def test_reserved_separators_are_refused(bad):
    with pytest.raises(ValueError, match='reserve'):
        connector_token(bad, 'm')
    with pytest.raises(ValueError, match='reserve'):
        connector_token('c', bad)
