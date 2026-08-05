import numpy as np
import pytest

from nnphysics.core.errors import UnknownNameError, ValidationError
from nnphysics.core.types import FloatArray, Regime
from nnphysics.systems.fluid import (
    FLUID_REGIMES,
    FluidDynamics,
    FluidGrid,
    characteristic_length,
    initial_state,
    taylor_green_decay_rate,
    taylor_green_vorticity,
    viscosity,
)

GRID = FluidGrid(32)
DYNAMICS = FluidDynamics(GRID, 0.001)


def _shell_spectrum(vorticity: FloatArray) -> FloatArray:
    """Energy against integer wavenumber, summed over each shell."""
    dynamics = FluidDynamics(GRID)
    magnitude = np.sqrt(dynamics.operators.squared_wavenumber)
    spectrum = GRID.forward(vorticity) * dynamics.operators.inverse_squared_wavenumber
    energy = np.abs(spectrum) ** 2 * dynamics.operators.squared_wavenumber
    # The half spectrum counts every mode once except those on the two real columns.
    weight = np.full(spectrum.shape, 2.0)
    weight[:, 0] = 1.0
    shells = np.bincount(
        np.rint(magnitude * characteristic_length(GRID)).astype(np.int64).ravel(),
        weights=(energy * weight).ravel(),
        minlength=GRID.size,
    )
    return np.asarray(shells, dtype=np.float64)


class TestViscosityFromReynolds:
    def test_the_default_domain_makes_it_the_inverse_reynolds_number(self) -> None:
        regime = Regime("taylor_green", {"reynolds": 250.0})
        assert viscosity(regime, FluidGrid(32)) == pytest.approx(1.0 / 250.0)

    def test_it_scales_with_the_domain(self) -> None:
        regime = Regime("taylor_green", {"reynolds": 100.0})
        assert viscosity(regime, FluidGrid(32, 1.0)) == pytest.approx(1.0 / (100.0 * 2.0 * np.pi))

    def test_an_infinite_reynolds_number_is_the_inviscid_limit(self) -> None:
        """Spelled as a limit rather than as a separate flag, because that is what it is."""
        regime = Regime("taylor_green", {"reynolds": float("inf")})
        assert viscosity(regime, GRID) == 0.0

    @pytest.mark.parametrize("reynolds", [0.0, -10.0, float("nan")])
    def test_a_reynolds_number_that_is_not_positive_is_rejected(self, reynolds: float) -> None:
        with pytest.raises(ValidationError, match="Reynolds"):
            viscosity(Regime("taylor_green", {"reynolds": reynolds}), GRID)

    def test_a_missing_reynolds_number_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires the parameter"):
            viscosity(Regime("taylor_green", {}), GRID)


class TestTaylorGreenField:
    def test_it_is_an_eigenfunction_of_the_laplacian(self) -> None:
        vorticity = taylor_green_vorticity(GRID, 3)
        streamfunction = GRID.inverse(DYNAMICS.streamfunction_spectrum(GRID.forward(vorticity)))

        assert streamfunction == pytest.approx(vorticity / (2.0 * 3.0**2))

    def test_the_phase_moves_the_pattern(self) -> None:
        shifted = taylor_green_vorticity(GRID, 1, (np.pi, 0.0))
        assert shifted == pytest.approx(-taylor_green_vorticity(GRID, 1))

    def test_the_decay_rate_is_set_by_the_squared_wavenumber(self) -> None:
        assert taylor_green_decay_rate(GRID, 3, 0.1) == pytest.approx(2.0 * 0.1 * 9.0)


class TestRegimes:
    def test_three_are_declared(self) -> None:
        assert [regime.name for regime in FLUID_REGIMES] == [
            "taylor_green",
            "decaying_turbulence",
            "shear_layer",
        ]

    @pytest.mark.parametrize("regime", FLUID_REGIMES, ids=lambda regime: regime.name)
    def test_every_regime_is_generated_with_unit_speed_and_no_mean(self, regime: Regime) -> None:
        """Unit speed is what gives the Reynolds number a meaning across regimes."""
        state = initial_state(regime, np.random.default_rng(0), DYNAMICS)
        vorticity = GRID.unpack(state)

        assert DYNAMICS.energy(vorticity) == pytest.approx(0.5)
        assert float(np.mean(vorticity)) == pytest.approx(0.0, abs=1e-12)
        assert state.time == 0.0

    @pytest.mark.parametrize("regime", FLUID_REGIMES, ids=lambda regime: regime.name)
    def test_every_regime_is_band_limited(self, regime: Regime) -> None:
        """Anything above the cutoff would sit in the state inert, never advected."""
        state = initial_state(regime, np.random.default_rng(0), DYNAMICS)
        spectrum = GRID.forward(GRID.unpack(state))

        assert np.abs(spectrum * ~DYNAMICS.operators.retained).max() < 1e-12

    @pytest.mark.parametrize("regime", FLUID_REGIMES, ids=lambda regime: regime.name)
    def test_a_seed_determines_the_field(self, regime: Regime) -> None:
        first = initial_state(regime, np.random.default_rng(11), DYNAMICS)
        again = initial_state(regime, np.random.default_rng(11), DYNAMICS)

        assert np.array_equal(GRID.unpack(first), GRID.unpack(again))

    @pytest.mark.parametrize("regime", FLUID_REGIMES, ids=lambda regime: regime.name)
    def test_another_seed_gives_another_field(self, regime: Regime) -> None:
        first = initial_state(regime, np.random.default_rng(11), DYNAMICS)
        other = initial_state(regime, np.random.default_rng(12), DYNAMICS)

        assert not np.allclose(GRID.unpack(first), GRID.unpack(other))

    def test_an_unknown_regime_is_rejected(self) -> None:
        regime = Regime("von_karman", {"reynolds": 100.0})
        with pytest.raises(UnknownNameError, match="unknown fluid regime"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)

    def test_an_unknown_parameter_is_rejected(self) -> None:
        regime = Regime("taylor_green", {"reynolds": 100.0, "amplitude": 2.0})
        with pytest.raises(ValidationError, match="unknown fluid regime"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)

    @pytest.mark.parametrize("wavenumber", [0.0, 11.0])
    def test_a_wavenumber_outside_the_resolved_band_is_rejected(self, wavenumber: float) -> None:
        """Eleven is above the cutoff of a thirty two point grid, so it would be truncated."""
        regime = Regime("taylor_green", {"reynolds": 100.0, "wavenumber": wavenumber})
        with pytest.raises(ValidationError, match="dealiasing cutoff"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)

    @pytest.mark.parametrize("width", [0.0, 0.5])
    def test_an_unusable_layer_width_is_rejected(self, width: float) -> None:
        regime = Regime("shear_layer", {"reynolds": 100.0, "layer_width": width})
        with pytest.raises(ValidationError, match="layer width"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)

    def test_a_layer_thinner_than_a_cell_is_rejected(self) -> None:
        """It would be a step function on this grid, and the spectrum would tell no one."""
        regime = Regime("shear_layer", {"reynolds": 100.0, "layer_width": 0.005})
        with pytest.raises(ValidationError, match="thinner than a cell"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)

    def test_a_perturbation_outside_the_unit_interval_is_rejected(self) -> None:
        regime = Regime("shear_layer", {"reynolds": 100.0, "perturbation": 1.5})
        with pytest.raises(ValidationError, match="perturbation"):
            initial_state(regime, np.random.default_rng(0), DYNAMICS)


class TestPrescribedSpectrum:
    def test_the_turbulence_spectrum_peaks_where_it_was_asked_to(self) -> None:
        regime = Regime("decaying_turbulence", {"reynolds": 500.0, "peak_wavenumber": 8.0})

        state = initial_state(regime, np.random.default_rng(4), DYNAMICS)
        shells = _shell_spectrum(GRID.unpack(state))

        assert abs(int(np.argmax(shells)) - 8) <= 2

    def test_moving_the_peak_moves_the_spectrum(self) -> None:
        low = initial_state(
            Regime("decaying_turbulence", {"reynolds": 500.0, "peak_wavenumber": 3.0}),
            np.random.default_rng(4),
            DYNAMICS,
        )
        high = initial_state(
            Regime("decaying_turbulence", {"reynolds": 500.0, "peak_wavenumber": 9.0}),
            np.random.default_rng(4),
            DYNAMICS,
        )

        assert int(np.argmax(_shell_spectrum(GRID.unpack(low)))) < int(
            np.argmax(_shell_spectrum(GRID.unpack(high)))
        )


class TestShearLayer:
    def test_without_a_perturbation_the_flow_is_parallel(self) -> None:
        regime = Regime("shear_layer", {"reynolds": 100.0, "perturbation": 0.0})

        vorticity = GRID.unpack(initial_state(regime, np.random.default_rng(0), DYNAMICS))

        assert vorticity == pytest.approx(np.tile(vorticity[0], (GRID.size, 1)))

    def test_the_two_layers_have_opposite_sign(self) -> None:
        regime = Regime("shear_layer", {"reynolds": 100.0, "perturbation": 0.0})

        vorticity = GRID.unpack(initial_state(regime, np.random.default_rng(0), DYNAMICS))
        profile = vorticity[0]

        assert profile[GRID.size // 4] * profile[3 * GRID.size // 4] < 0.0
