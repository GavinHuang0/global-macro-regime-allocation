from regime_allocation.types import MacroRegime


def test_regime_enum_has_four_states() -> None:
    assert len(MacroRegime) == 4
