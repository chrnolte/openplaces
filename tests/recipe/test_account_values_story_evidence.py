"""The parcel curate weighs buildings by the story evidence the building
curate decides from, in the same order (maintainer, 2026-11-18).

`divide_account_values` cannot read the decided `n_stories`: the
building curate runs after the parcel curate and reads it. It applies
`decide_stories`' evidence lanes itself, without the roll lane, so the
two recipe lists have to stay the same.
"""

from openplaces.recipe import get_recipe_by_id


def _step(recipe_id, name):
    steps = [
        s
        for s in get_recipe_by_id(recipe_id)['pipeline']
        if isinstance(s, dict) and s.get('step') == name
    ]
    assert len(steps) == 1, f'{recipe_id} has {len(steps)} {name} steps'
    return steps[0]


def test_parcel_story_evidence_follows_decide_stories():
    decide = _step('US_building-openplaces-2026', 'decide_stories')
    divide = _step('US_parcel-openplaces-2026', 'divide_account_values')
    # A measured height is a lane of decide_stories under the column it
    # converts into.
    converted = {decide.get('height_column'): decide.get('height_stories_column')}
    lanes = []
    for entry in divide['stories']['evidence']:
        name = entry.get('column') or converted.get(entry.get('height_column'))
        # One lane may be read from several recipes (CHEER per state).
        if name and (not lanes or lanes[-1] != name):
            lanes.append(name)
    assert lanes == list(decide['evidence']), (
        'The story evidence of divide_account_values (US_parcel-openplaces-'
        '2026) and of decide_stories (US_building-openplaces-2026) differ: '
        'change both lists together.'
    )
    assert divide['stories'].get('floor_height_m') == decide.get('floor_height_m')
    assert list(divide['stories'].get('valid_range')) == list(decide.get('valid_range'))
