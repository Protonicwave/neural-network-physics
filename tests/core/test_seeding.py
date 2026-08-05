import numpy as np
import pytest
import torch

from nnphysics.core.seeding import (
    make_deterministic,
    numpy_generator,
    seed_sequence,
    torch_generator,
)


def test_numpy_stream_is_reproducible() -> None:
    first = numpy_generator(0, "data.train").standard_normal(8)
    second = numpy_generator(0, "data.train").standard_normal(8)
    assert np.array_equal(first, second)


def test_numpy_streams_are_independent_of_each_other() -> None:
    train = numpy_generator(0, "data.train").standard_normal(8)
    validation = numpy_generator(0, "data.val").standard_normal(8)
    assert not np.array_equal(train, validation)


def test_numpy_stream_depends_on_the_root_seed() -> None:
    first = numpy_generator(0, "data.train").standard_normal(8)
    second = numpy_generator(1, "data.train").standard_normal(8)
    assert not np.array_equal(first, second)


def test_torch_stream_is_reproducible() -> None:
    first = torch.randn(8, generator=torch_generator(0, "model.init"))
    second = torch.randn(8, generator=torch_generator(0, "model.init"))
    assert torch.equal(first, second)


def test_torch_streams_are_independent_of_each_other() -> None:
    init = torch.randn(8, generator=torch_generator(0, "model.init"))
    dropout = torch.randn(8, generator=torch_generator(0, "model.dropout"))
    assert not torch.equal(init, dropout)


def test_numpy_and_torch_streams_of_one_name_do_not_track_each_other() -> None:
    from_numpy = numpy_generator(0, "shared").random(8)
    from_torch = torch.rand(8, generator=torch_generator(0, "shared")).numpy()
    assert not np.allclose(from_numpy, from_torch)


def test_a_generator_is_fresh_each_time_it_is_asked_for() -> None:
    generator = numpy_generator(0, "data.train")
    first = generator.standard_normal(8)
    assert not np.array_equal(first, generator.standard_normal(8))
    assert np.array_equal(first, numpy_generator(0, "data.train").standard_normal(8))


def test_empty_stream_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="stream name"):
        seed_sequence(0, "")


def test_negative_root_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non negative"):
        seed_sequence(-1, "data.train")
    with pytest.raises(ValueError, match="non negative"):
        make_deterministic(-1)


def test_make_deterministic_pins_the_global_generators() -> None:
    make_deterministic(3)
    first = torch.randn(4)
    make_deterministic(3)
    assert torch.equal(first, torch.randn(4))
