"""Evaluation and analysis.

Exposes counting metrics, L0-L4 tier evaluation, error-source decomposition and
independent annotation parsing. The visualization module depends on matplotlib and is
imported on demand so headless environments are unaffected.
"""

from .error_analysis import (
    counting_breakdown,
    decompose_with_support,
    density_mass_decomposition,
    image_error_record,
    masking_effect_record,
    select_representative_cases,
    summarize_error_sources,
    summarize_masking_effect,
)
from .labelbox import build_index, load_index, load_split, parse_record, split_counts
from .metrics import (
    apply_output_mask,
    check_count_preservation,
    count_from_density,
    counts_from_batch,
    density_counts,
    mae,
    relative_count_error,
    rmse,
    summarize_counts,
)
from .tier_eval import (
    TIER_NAMES,
    assign_tier,
    evaluate_by_tier,
    tier_distribution,
)

__all__ = [
    "TIER_NAMES",
    "apply_output_mask",
    "assign_tier",
    "build_index",
    "check_count_preservation",
    "count_from_density",
    "counts_from_batch",
    "counting_breakdown",
    "decompose_with_support",
    "density_counts",
    "density_mass_decomposition",
    "evaluate_by_tier",
    "image_error_record",
    "load_index",
    "load_split",
    "mae",
    "masking_effect_record",
    "parse_record",
    "relative_count_error",
    "rmse",
    "select_representative_cases",
    "split_counts",
    "summarize_counts",
    "summarize_error_sources",
    "summarize_masking_effect",
    "tier_distribution",
]
