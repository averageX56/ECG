from ecg_project.processing.parallel import ordered_map
import pytest


def test_spawn_preserves_order_and_propagates_errors():
    assert list(ordered_map(abs,[-3,1,-2,0],workers=2))==[3,1,2,0]
    with pytest.raises(TypeError):list(ordered_map(abs,[None],workers=2))
    with pytest.raises(ValueError):list(ordered_map(abs,[1],workers=0))
