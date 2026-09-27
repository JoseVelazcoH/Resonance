from mood_dj.domain.app_ownership import APP_DESCRIPTION_MARKER, is_app_created


def test_description_marker_uses_the_current_brand() -> None:
    assert APP_DESCRIPTION_MARKER == "Created by Resonance"


def test_not_app_created_when_no_marker_and_unrelated_name() -> None:
    assert is_app_created("Road Trip", "My favorite driving songs") is False


def test_app_created_when_description_has_current_marker() -> None:
    assert is_app_created("Road Trip", "Created by Resonance from a mood prompt") is True


def test_app_created_when_description_has_legacy_marker() -> None:
    # Playlists saved before the rename must never flow back into the library.
    assert is_app_created("Road Trip", "Created by Moodify from a mood prompt") is True


def test_app_created_when_name_uses_current_brand_prefix() -> None:
    assert is_app_created("Resonance · Happy · Sep 27", None) is True


def test_app_created_when_name_uses_legacy_brand_prefix() -> None:
    assert is_app_created("Moodify - Estoy triste", None) is True
    assert is_app_created("Moodify · Estoy triste · Sep 26", None) is True


def test_brand_prefix_is_case_insensitive() -> None:
    assert is_app_created("RESONANCE - test", None) is True


def test_name_starting_with_brand_but_no_separator_is_not_app_created() -> None:
    assert is_app_created("Resonanceish Vibes", None) is False
    assert is_app_created("Moodifyish Vibes", None) is False


def test_empty_description_is_not_a_marker() -> None:
    assert is_app_created("Road Trip", "") is False
