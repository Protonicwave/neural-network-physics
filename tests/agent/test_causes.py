from __future__ import annotations

from nnphysics.agent.causes import CAUSE_DESCRIPTIONS, Cause, cause_catalogue, describe_cause
from nnphysics.agent.faults import FAULTS

DISTRACTORS = 5
"""Causes no fault uses. Without them the vocabulary would be the answer key, and a
diagnoser that guessed uniformly would be hard to tell from one that read the numbers."""


class TestVocabulary:
    def test_every_cause_is_described(self) -> None:
        """A diagnoser choosing from bare identifiers is guessing at what they mean."""
        assert set(CAUSE_DESCRIPTIONS) == set(Cause)

    def test_no_description_is_empty(self) -> None:
        for cause in Cause:
            assert len(describe_cause(cause)) > 40

    def test_the_catalogue_lists_every_cause_once(self) -> None:
        catalogue = cause_catalogue()

        for cause in Cause:
            assert catalogue.count(f"- {cause.value}:") == 1

    def test_there_are_more_causes_than_faults(self) -> None:
        used = {entry.cause for entry in FAULTS}

        assert len(set(Cause) - used) == DISTRACTORS

    def test_the_values_are_stable_identifiers(self) -> None:
        """Written into every committed score, so they are part of the artefact."""
        assert Cause.ROLLOUT_CURRICULUM.value == "rollout_curriculum"
        assert all(cause.value.islower() for cause in Cause)
