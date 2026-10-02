"""
Score a curated classification against hand-labeled ground-truth points.

Vocabulary-neutral and geography-neutral by construction: no class names, no
admin units, and no source paths appear here. Callers supply the labelled
points, the curated entities, and the class list, so the same code validates
building occupancy in North Carolina, land use elsewhere, or any future
labelled set (a state manufactured-housing registry, building permits) without
being edited.

Two things this module insists on that a naive accuracy check gets wrong:

- **Identity beats proximity when linking.** The nearest footprint to a survey
  pin is very often a shed or the neighbor's house, so an address match is
  tried first and distance is only the fallback. Which route matched is
  recorded, because a distance-linked row is weaker evidence than an
  address-linked one and the difference should stay visible downstream.
  Identity alone does not finish the job, though: a house and its garage
  share one address, so several entities routinely match the same point.
  Callers rank those ties with `prefer_column`; without it the pick falls to
  row order, which is arbitrary.
- **Precision and recall are reported separately, per class.** A rule that
  labels almost everything one class scores excellent recall for it, and an
  aggregate agreement figure hides the whole problem: it is entirely possible
  for overall agreement to rise while two of three classes get worse.
- **The full confusion matrix is the default output.** Every scoring
  path builds one (:func:`confusion_matrix`, reference rows, predicted
  columns, abstentions and off-vocabulary predictions kept) and derives
  producer's and consumer's accuracy from it
  (:func:`accuracy_from_matrix`); :func:`write_confusion_report` writes
  both, aggregates only, into a delivery's accuracies folder.

One package since 2026-09-29 (until then a single 2,127-line module):
`linking` (points to entities), `accuracy` (matrices, kappa, year
agreement), `reports` (the aggregate files), `references` (reference
classes and tiers) and `context` (`ValidationContext`); every name is
re-exported here, so imports from `openplaces.io.curator.validation`
keep working.
"""

from __future__ import annotations

from openplaces.io.curator.validation.accuracy import (  # noqa: F401
    ABSTAIN_LABEL,
    ACCURACY_COLUMNS,
    MIN_STRATUM_ROWS,
    NO_REFERENCE_LABEL,
    NON_RESIDENTIAL_LABEL,
    OTHER_LABEL,
    YEAR_AGREEMENT_BINS,
    YEAR_REFERENCE_LABEL,
    _class_counts,
    _f1,
    _matrix_classes,
    _ratio,
    accuracy_from_matrix,
    bin_year_agreement,
    cohens_kappa,
    compare_classifications_paired,
    confusion_matrix,
    paired_disagreement,
    score_classification,
    year_error_summary,
)
from openplaces.io.curator.validation.context import (  # noqa: F401
    ValidationContext,
    validation_context,
    validation_notebooks,
)
from openplaces.io.curator.validation.linking import (  # noqa: F401
    RESULT_COMMISSION,
    RESULT_CORRECT,
    RESULT_OMISSION,
    classify_validation_result,
    link_points_to_entities,
    normalize_house_number,
    summarize_sources,
)
from openplaces.io.curator.validation.references import (  # noqa: F401
    class_from_ruleset,
    reference_confidence_tier,
)
from openplaces.io.curator.validation.reports import (  # noqa: F401
    _as_strata,
    _long_form,
    _positional,
    _stratum_masks,
    _write_report,
    read_confusion_matrix,
    write_confusion_report,
    write_year_agreement_report,
)
