"""Short, recognizable codes for administrative-unit identifiers.

An admin id such as US-MA-SOM is read by people navigating directories and
filenames, so each segment has to be recognizable from the unit's name at a
glance. This package generates those segments in two steps:

1. :func:`~openplaces.io.admin_codes.candidates.generate_candidates` proposes
   a preference-ordered list of codes for one name, using a per-language
   vocabulary of articles, prepositions, conjunctions and qualifiers.
2. :func:`~openplaces.io.admin_codes.assign.assign_codes` picks one code per
   unit across a group of siblings by maximum-weight bipartite matching, so
   the outcome is optimal and independent of row order.

Separating the two matters. The predecessor implementation,
:func:`openplaces.io.admin.generate_admin_ids`, interleaved them in a
priority waterfall that assigned whichever code happened to be free first,
making its output depend on input ordering.
"""

from openplaces.io.admin_codes.anchors import (
    get_anchor_codes,
    get_code_length_convention,
    load_code_conventions,
    load_prior_codes,
    normalize_name,
)
from openplaces.io.admin_codes.assign import assign_codes, rank_score
from openplaces.io.admin_codes.audit import (
    audit_spine,
    resolve_identifier,
)
from openplaces.io.admin_codes.candidates import (
    CODE_PATTERN,
    Candidate,
    generate_candidates,
    is_valid_code,
    tokenize,
)
from openplaces.io.admin_codes.coverage import (
    DEFAULT_MIN_COVERAGE,
    intuitive_codes,
    intuitive_coverage,
    recommend_code_length,
    syllable_onsets,
)
from openplaces.io.admin_codes.derive import derive_codes
from openplaces.io.admin_codes.frame import assign_admin_ids
from openplaces.io.admin_codes.languages import (
    LanguagePack,
    fold_diacritics,
    get_language_pack,
    load_country_languages,
    load_language_packs,
)
from openplaces.io.admin_codes.registry import (
    load_registry,
    spine_path,
)

__all__ = [
    'CODE_PATTERN',
    'DEFAULT_MIN_COVERAGE',
    'Candidate',
    'LanguagePack',
    'assign_admin_ids',
    'audit_spine',
    'resolve_identifier',
    'assign_codes',
    'derive_codes',
    'intuitive_codes',
    'intuitive_coverage',
    'recommend_code_length',
    'syllable_onsets',
    'fold_diacritics',
    'generate_candidates',
    'get_anchor_codes',
    'get_code_length_convention',
    'get_language_pack',
    'is_valid_code',
    'load_code_conventions',
    'load_country_languages',
    'load_language_packs',
    'load_prior_codes',
    'load_registry',
    'spine_path',
    'normalize_name',
    'rank_score',
    'tokenize',
]
