import numpy as np
import pytest

from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.core.types import Regime, State
from nnphysics.systems.nbody import NBODY_REGIMES, NBodyDynamics, TotalEnergy, initial_state, unpack
from nnphysics.systems.nbody.dynamics import kinetic_energy

DYNAMICS = NBodyDynamics(gravitational_constant=1.0, softening=0.01)
REGIMES_BY_NAME = {regime.name: regime for regime in NBODY_REGIMES}


def draw(name: str, seed: int = 0) -> State:
    return initial_state(REGIMES_BY_NAME[name], np.random.default_rng(seed), DYNAMICS)


@pytest.mark.parametrize("regime", NBODY_REGIMES, ids=lambda item: item.name)
class TestEveryRegime:
    def test_it_produces_a_valid_state_at_time_zero(self, regime: Regime) -> None:
        state = initial_state(regime, np.random.default_rng(0), DYNAMICS)
        state.require_finite()
        assert state.time == 0.0
        assert unpack(state)[0].shape[0] >= 2

    def test_the_total_mass_is_one(self, regime: Regime) -> None:
        _, _, mass = unpack(initial_state(regime, np.random.default_rng(1), DYNAMICS))
        assert float(np.sum(mass)) == pytest.approx(1.0)

    def test_the_centre_of_mass_is_at_the_origin_and_at_rest(self, regime: Regime) -> None:
        position, velocity, mass = unpack(initial_state(regime, np.random.default_rng(2), DYNAMICS))
        weights = mass[:, np.newaxis]

        assert np.allclose(np.sum(weights * position, axis=0), 0.0, atol=1e-14)
        assert np.allclose(np.sum(weights * velocity, axis=0), 0.0, atol=1e-14)

    def test_the_same_seed_gives_a_bitwise_identical_state(self, regime: Regime) -> None:
        first = initial_state(regime, np.random.default_rng(3), DYNAMICS)
        second = initial_state(regime, np.random.default_rng(3), DYNAMICS)

        for name in first.names:
            assert np.array_equal(first.fields[name], second.fields[name])

    def test_different_seeds_give_different_states(self, regime: Regime) -> None:
        first = initial_state(regime, np.random.default_rng(4), DYNAMICS)
        second = initial_state(regime, np.random.default_rng(5), DYNAMICS)

        assert not np.array_equal(first.fields["position"], second.fields["position"])


class TestColdCollapse:
    def test_it_starts_at_rest(self) -> None:
        _, velocity, _ = unpack(draw("cold_collapse"))
        assert np.array_equal(velocity, np.zeros_like(velocity))

    def test_it_is_bound(self) -> None:
        assert TotalEnergy(DYNAMICS).evaluate(draw("cold_collapse")) < 0.0


class TestVirialisedCluster:
    def test_the_virial_ratio_is_met(self) -> None:
        position, velocity, mass = unpack(draw("virialised_cluster"))
        ratio = (
            2.0 * kinetic_energy(velocity, mass) / abs(DYNAMICS.potential_energy(position, mass))
        )
        assert ratio == pytest.approx(1.0)

    def test_the_masses_span_the_declared_spread(self) -> None:
        _, _, mass = unpack(draw("virialised_cluster", seed=11))
        assert float(np.max(mass) / np.min(mass)) > 1.5


class TestHierarchicalPair:
    def test_it_has_two_tight_binaries_and_a_wide_separation(self) -> None:
        position, _, _ = unpack(draw("hierarchical_pair", seed=6))
        assert position.shape[0] == 4

        inner = [
            float(np.linalg.norm(position[0] - position[1])),
            float(np.linalg.norm(position[2] - position[3])),
        ]
        outer = float(np.linalg.norm(0.5 * (position[0] + position[1] - position[2] - position[3])))

        assert all(value == pytest.approx(0.2) for value in inner)
        assert outer == pytest.approx(2.0)


class TestValidation:
    def test_an_unknown_regime_is_rejected(self) -> None:
        with pytest.raises(UnknownNameError, match="unknown nbody regime"):
            initial_state(Regime("turbulent", {}), np.random.default_rng(0), DYNAMICS)

    def test_an_unknown_parameter_is_rejected(self) -> None:
        regime = Regime("cold_collapse", {"n_bodies": 8.0, "reynolds": 100.0})
        with pytest.raises(ValidationError, match="unknown"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)

    def test_a_fractional_body_count_is_rejected(self) -> None:
        regime = Regime("cold_collapse", {"n_bodies": 8.5})
        with pytest.raises(ValidationError, match="whole number"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)

    @pytest.mark.parametrize(
        ("parameters", "message"),
        [
            ({"n_bodies": 1.0}, "at least two bodies"),
            ({"radius": -1.0}, "radius must be positive"),
            ({"mass_spread": 0.5}, "mass spread"),
            ({"virial_ratio": -0.5}, "virial ratio"),
        ],
    )
    def test_an_out_of_range_cluster_parameter_is_rejected(
        self, parameters: dict[str, float], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            initial_state(Regime("cold_collapse", parameters), np.random.default_rng(0), DYNAMICS)

    def test_a_pair_that_is_not_hierarchical_is_rejected(self) -> None:
        regime = Regime("hierarchical_pair", {"inner_separation": 1.0, "outer_separation": 2.0})
        with pytest.raises(ValidationError, match="only hierarchical if"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)
