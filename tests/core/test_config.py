from collections.abc import Mapping
from pathlib import Path

import pytest

from nnphysics.core.config import load_run_config
from nnphysics.core.errors import ConfigurationError
from nnphysics.data.layout import dataset_id

REPO_ROOT = Path(__file__).resolve().parents[2]

MINIMAL = """
name: unit
seed: 7
system:
  name: placeholder
data:
  regimes: [a]
  held_out_regimes: [b]
model:
  name: placeholder
evaluation:
  metrics: [placeholder]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_minimal_config_loads_and_is_frozen(tmp_path: Path) -> None:
    config = load_run_config(write(tmp_path, MINIMAL))
    assert config.name == "unit"
    assert config.seed == 7
    assert config.training.epochs == 50
    with pytest.raises(ValueError, match="frozen"):
        config.name = "other"


def test_shipped_example_config_is_valid() -> None:
    config = load_run_config(REPO_ROOT / "configs" / "example.yaml")
    assert config.name == "example"


def test_missing_file_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read configuration"):
        load_run_config(tmp_path / "absent.yaml")


def test_malformed_yaml_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        load_run_config(write(tmp_path, "name: [unclosed\n"))


def test_non_mapping_document_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="mapping at the top level"):
        load_run_config(write(tmp_path, "- one\n- two\n"))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="invalid configuration"):
        load_run_config(write(tmp_path, MINIMAL + "nonsense: 1\n"))


def test_missing_required_section_is_rejected(tmp_path: Path) -> None:
    without_model = MINIMAL.replace("model:\n  name: placeholder\n", "")
    with pytest.raises(ConfigurationError, match="invalid configuration"):
        load_run_config(write(tmp_path, without_model))


def test_negative_seed_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="invalid configuration"):
        load_run_config(write(tmp_path, MINIMAL.replace("seed: 7", "seed: -1")))


def test_splits_must_leave_a_training_split(tmp_path: Path) -> None:
    greedy = MINIMAL.replace(
        "  held_out_regimes: [b]",
        "  held_out_regimes: [b]\n  val_fraction: 0.6\n  test_fraction: 0.5",
    )
    with pytest.raises(ConfigurationError, match="non empty training split"):
        load_run_config(write(tmp_path, greedy))


def test_held_out_regime_may_not_also_be_trained_on(tmp_path: Path) -> None:
    overlapping = MINIMAL.replace("held_out_regimes: [b]", "held_out_regimes: [a]")
    with pytest.raises(ConfigurationError, match="both trained on and held out"):
        load_run_config(write(tmp_path, overlapping))


def test_evaluation_needs_at_least_one_metric(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="invalid configuration"):
        load_run_config(write(tmp_path, MINIMAL.replace("metrics: [placeholder]", "metrics: []")))


def test_a_suite_carries_the_settings_its_numbers_depend_on(tmp_path: Path) -> None:
    """A metric list on its own is not a suite.

    The same metric over a different horizon answers a different question, so the settings
    belong to the name a result file records.
    """
    config = load_run_config(write(tmp_path, MINIMAL))

    assert config.evaluation.name
    assert config.evaluation.predictors
    assert config.evaluation.rollout_steps > 0
    assert config.evaluation.error_thresholds
    assert 0.0 < config.evaluation.distribution_window < 1.0


@pytest.mark.parametrize("field", ["metrics", "predictors"])
def test_a_suite_may_not_name_the_same_entry_twice(tmp_path: Path, field: str) -> None:
    """Two entries would be averaged into one column, which reads as one run of it."""
    suite = (
        "  metrics: [a, a]"
        if field == "metrics"
        else "  metrics: [placeholder]\n  predictors: [a, a]"
    )
    doubled = MINIMAL.replace("  metrics: [placeholder]", suite)
    with pytest.raises(ConfigurationError, match="more than once"):
        load_run_config(write(tmp_path, doubled))


def test_environment_overrides_are_applied_and_typed(tmp_path: Path) -> None:
    env: Mapping[str, str] = {
        "NNP_SEED": "11",
        "NNP_TRAINING__EPOCHS": "3",
        "NNP_TRAINING__LEARNING_RATE": "0.05",
        "PATH": "ignored",
    }
    config = load_run_config(write(tmp_path, MINIMAL), env)
    assert config.seed == 11
    assert config.training.epochs == 3
    assert config.training.learning_rate == pytest.approx(0.05)


def test_environment_is_ignored_when_not_passed(tmp_path: Path) -> None:
    config = load_run_config(write(tmp_path, MINIMAL), None)
    assert config.seed == 7


def test_override_through_a_non_mapping_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot descend into"):
        load_run_config(write(tmp_path, MINIMAL), {"NNP_NAME__DEEPER": "1"})


def test_malformed_override_name_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="malformed configuration override"):
        load_run_config(write(tmp_path, MINIMAL), {"NNP_TRAINING__": "1"})


def test_run_id_is_stable_for_the_same_inputs(tmp_path: Path) -> None:
    first = load_run_config(write(tmp_path, MINIMAL))
    second = load_run_config(write(tmp_path, MINIMAL))
    assert first.run_id == second.run_id
    assert len(first.run_id) == 16


def test_run_id_is_insensitive_to_key_order(tmp_path: Path) -> None:
    reordered = "\n".join(
        [
            "seed: 7",
            "name: unit",
            "model:",
            "  name: placeholder",
            "evaluation:",
            "  metrics: [placeholder]",
            "system:",
            "  name: placeholder",
            "data:",
            "  regimes: [a]",
            "  held_out_regimes: [b]",
        ]
    )
    baseline = load_run_config(write(tmp_path, MINIMAL))
    assert load_run_config(write(tmp_path, reordered)).run_id == baseline.run_id


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("seed: 7", "seed: 8"),
        ("name: unit", "name: other"),
        ("  regimes: [a]", "  regimes: [a, c]"),
    ],
)
def test_run_id_changes_when_any_input_changes(tmp_path: Path, old: str, new: str) -> None:
    baseline = load_run_config(write(tmp_path, MINIMAL))
    changed = load_run_config(write(tmp_path, MINIMAL.replace(old, new)))
    assert baseline.run_id != changed.run_id


def test_run_id_changes_when_a_default_is_set_explicitly_to_something_else(
    tmp_path: Path,
) -> None:
    baseline = load_run_config(write(tmp_path, MINIMAL))
    overridden = load_run_config(write(tmp_path, MINIMAL), {"NNP_TRAINING__EPOCHS": "51"})
    assert baseline.run_id != overridden.run_id


def test_run_dir_carries_the_name_and_the_run_id(tmp_path: Path) -> None:
    config = load_run_config(write(tmp_path, MINIMAL))
    assert config.run_dir == Path("runs") / f"unit-{config.run_id}"


class TestEnsembleMembers:
    """The rule an ensemble's spread means anything under.

    Members differ in their initialisation and not in their data.
    """

    def test_member_zero_is_the_plain_run(self, tmp_path: Path) -> None:
        config = load_run_config(write(tmp_path, MINIMAL))

        assert config.member == 0
        assert config.run_seed == config.seed
        assert config.for_member(0) == config

    def test_a_later_member_draws_from_a_different_seed(self, tmp_path: Path) -> None:
        config = load_run_config(write(tmp_path, MINIMAL))

        seeds = {config.for_member(index).run_seed for index in range(config.ensemble.members)}

        assert len(seeds) == config.ensemble.members

    def test_the_member_seed_is_a_function_of_the_run_seed_and_the_index(
        self, tmp_path: Path
    ) -> None:
        """Derived rather than drawn.

        The same configuration gives the same members however many times it is resolved.
        """
        first = load_run_config(write(tmp_path, MINIMAL))
        again = load_run_config(write(tmp_path, MINIMAL))

        assert first.for_member(2).run_seed == again.for_member(2).run_seed

    def test_changing_the_root_seed_moves_every_member(self, tmp_path: Path) -> None:
        config = load_run_config(write(tmp_path, MINIMAL))
        other = load_run_config(write(tmp_path, MINIMAL.replace("seed: 7", "seed: 8")))

        assert config.for_member(1).run_seed != other.for_member(1).run_seed

    def test_the_dataset_does_not_depend_on_the_member(self, tmp_path: Path) -> None:
        """The rule the whole estimate rests on.

        Members trained on different data would be measuring the data rather than the
        initialisation, and their disagreement would mean nothing.
        """
        config = load_run_config(write(tmp_path, MINIMAL))
        identifiers = {
            dataset_id(config.for_member(index)) for index in range(config.ensemble.members)
        }

        assert len(identifiers) == 1

    def test_each_member_gets_its_own_run_identifier(self, tmp_path: Path) -> None:
        """Anything that changes the weights has to change the identifier.

        Two members that hashed alike would write over each other.
        """
        config = load_run_config(write(tmp_path, MINIMAL))

        identifiers = {config.for_member(index).run_id for index in range(config.ensemble.members)}

        assert len(identifiers) == config.ensemble.members

    @pytest.mark.parametrize("index", [-1, 4, 99])
    def test_a_member_the_ensemble_does_not_have_is_refused(
        self, tmp_path: Path, index: int
    ) -> None:
        config = load_run_config(write(tmp_path, MINIMAL))

        with pytest.raises(ConfigurationError, match="is not one of"):
            config.for_member(index)
