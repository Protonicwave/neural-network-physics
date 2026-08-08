from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nnphysics.reporting.index import KEY_SCALARS, index_runs, summarise
from nnphysics.reporting.layout import RECORD_NAME
from nnphysics.reporting.page import summarise_run
from nnphysics.reporting.record import RunRecord, write_record

Factory = Callable[..., RunRecord]


def store(root: Path, record: RunRecord) -> Path:
    directory = root / f"{record.name}-{record.run_id}"
    directory.mkdir(parents=True)
    write_record(directory / RECORD_NAME, record)
    return directory


class TestSummarise:
    def test_it_gives_one_row_per_predictor_and_split(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        summaries = summarise(make_record(), tmp_path)

        assert len(summaries) == 4
        assert {entry.split for entry in summaries} == {"test", "held_out"}

    def test_it_carries_the_key_scalars(self, make_record: Factory, tmp_path: Path) -> None:
        summary = summarise(make_record(), tmp_path)[0]

        assert set(summary.values) == {f"{metric}.{key}" for metric, key in KEY_SCALARS}

    def test_it_can_be_narrowed_to_one_predictor(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        summaries = summarise(make_record(), tmp_path, predictor="reference")

        assert {entry.predictor for entry in summaries} == {"reference"}

    def test_it_can_be_narrowed_to_one_split(self, make_record: Factory, tmp_path: Path) -> None:
        summaries = summarise(make_record(), tmp_path, split="test")

        assert {entry.split for entry in summaries} == {"test"}

    def test_it_carries_the_same_verdict_the_page_states(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        record = make_record()

        summary = summarise(record, tmp_path)[0]

        assert summary.verdict == summarise_run(record, tmp_path.name).verdict.phrase

    def test_it_carries_the_usable_stretch_the_page_derives(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        record = make_record()
        summary = summarise(record, tmp_path)[0]

        card = summarise_run(record, tmp_path.name)
        horizon = card.horizon(summary.predictor, summary.split)

        assert horizon is not None
        assert summary.usable == horizon.steps

    def test_it_labels_a_run_by_name_and_identifier(
        self, make_record: Factory, tmp_path: Path
    ) -> None:
        summary = summarise(make_record(name="nbody", run_id="abcd"), tmp_path)[0]

        assert summary.label == "nbody (abcd)"


class TestIndex:
    def test_an_empty_root_lists_nothing(self, tmp_path: Path) -> None:
        assert index_runs(tmp_path) == ()

    def test_it_reads_every_run_under_the_root(self, make_record: Factory, tmp_path: Path) -> None:
        store(tmp_path, make_record(run_id="aaaa"))
        store(tmp_path, make_record(run_id="bbbb"))

        summaries = index_runs(tmp_path, predictor="reference", split="test")

        assert {entry.run_id for entry in summaries} == {"aaaa", "bbbb"}

    def test_it_lists_the_oldest_run_first(self, make_record: Factory, tmp_path: Path) -> None:
        store(tmp_path, make_record(run_id="bbbb", created="2026-02-01T00:00:00+00:00"))
        store(tmp_path, make_record(run_id="aaaa", created="2026-01-01T00:00:00+00:00"))

        summaries = index_runs(tmp_path, predictor="reference", split="test")

        assert [entry.run_id for entry in summaries] == ["aaaa", "bbbb"]

    def test_it_says_where_each_run_lives(self, make_record: Factory, tmp_path: Path) -> None:
        directory = store(tmp_path, make_record(run_id="aaaa"))

        summary = index_runs(tmp_path, predictor="reference", split="test")[0]

        assert summary.directory == directory
