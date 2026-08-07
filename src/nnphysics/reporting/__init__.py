"""Reporting: run records, plots, reports and comparison between runs.

This layer imports `core`, `data` and `evals`. It never imports `systems` and never
imports `models`: a report describes a run through the record the run wrote, and a report
that had to build a system to describe one would be a report that could only describe the
systems it knew about.

The order things happen in is fixed. An evaluation writes a record. A record renders to a
document. A document serialises to Markdown and to a single self contained HTML file. Two
records compare. Nothing in that chain reads a clock or an environment: everything that
varies is captured once, at the edge, and written into the record.
"""

from nnphysics.reporting.compare import (
    DEFAULT_THRESHOLD,
    Comparison,
    Delta,
    Verdict,
    compare_records,
    compare_series,
)
from nnphysics.reporting.document import Document, to_html, to_markdown
from nnphysics.reporting.environment import (
    EnvironmentRecord,
    describe_environment,
    git_commit,
)
from nnphysics.reporting.explain import (
    TIMING_METRIC,
    Direction,
    Explanation,
    explain,
    explanations,
    metric_summary,
)
from nnphysics.reporting.index import KEY_SCALARS, RunSummary, index_runs, summarise
from nnphysics.reporting.layout import (
    HTML_NAME,
    MARKDOWN_NAME,
    PLOTS_DIR,
    RECORD_NAME,
    SNAPSHOTS_NAME,
    RunPaths,
    find_record,
    find_records,
    run_paths,
)
from nnphysics.reporting.page import (
    HARNESS_PREDICTORS,
    HELD_OUT_SPLIT,
    TRAINED_ON_SPLIT,
    USABLE_THRESHOLD,
    CostLadder,
    CostPoint,
    DiagnoserScore,
    Diagnosis,
    FaultRank,
    Headlines,
    Horizon,
    MatchedCost,
    PageModel,
    RunCard,
    RunVerdict,
    VerdictKind,
    build_page,
    read_fault_scores,
    summarise_run,
)
from nnphysics.reporting.plots import PlotRecord, overlay_plot, render_plots
from nnphysics.reporting.record import (
    RECORD_SCHEMA_VERSION,
    RunRecord,
    read_record,
    write_record,
)
from nnphysics.reporting.render import (
    build_comparison_document,
    build_document,
    render_comparison_html,
    render_comparison_markdown,
    render_html,
    render_markdown,
)
from nnphysics.reporting.theme import (
    DARK,
    LIGHT,
    TYPEFACES,
    DarkSwitch,
    Palette,
    Typefaces,
    css_var,
    report_stylesheet,
    token_names,
    tokens_css,
)

__all__ = [
    "DARK",
    "DEFAULT_THRESHOLD",
    "HARNESS_PREDICTORS",
    "HELD_OUT_SPLIT",
    "HTML_NAME",
    "KEY_SCALARS",
    "LIGHT",
    "MARKDOWN_NAME",
    "PLOTS_DIR",
    "RECORD_NAME",
    "RECORD_SCHEMA_VERSION",
    "SNAPSHOTS_NAME",
    "TIMING_METRIC",
    "TRAINED_ON_SPLIT",
    "TYPEFACES",
    "USABLE_THRESHOLD",
    "Comparison",
    "CostLadder",
    "CostPoint",
    "DarkSwitch",
    "Delta",
    "DiagnoserScore",
    "Diagnosis",
    "Direction",
    "Document",
    "EnvironmentRecord",
    "Explanation",
    "FaultRank",
    "Headlines",
    "Horizon",
    "MatchedCost",
    "PageModel",
    "Palette",
    "PlotRecord",
    "RunCard",
    "RunPaths",
    "RunRecord",
    "RunSummary",
    "RunVerdict",
    "Typefaces",
    "Verdict",
    "VerdictKind",
    "build_comparison_document",
    "build_document",
    "build_page",
    "compare_records",
    "compare_series",
    "css_var",
    "describe_environment",
    "explain",
    "explanations",
    "find_record",
    "find_records",
    "git_commit",
    "index_runs",
    "metric_summary",
    "overlay_plot",
    "read_fault_scores",
    "read_record",
    "render_comparison_html",
    "render_comparison_markdown",
    "render_html",
    "render_markdown",
    "render_plots",
    "report_stylesheet",
    "run_paths",
    "summarise",
    "summarise_run",
    "to_html",
    "to_markdown",
    "token_names",
    "tokens_css",
    "write_record",
]
