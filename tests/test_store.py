"""The card box: storing, finding again, surviving a restart."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from module_loader import load

store_module = load("store")
Store = store_module.Store
SOURCE_LEARNED = store_module.SOURCE_LEARNED
SOURCE_SEED = store_module.SOURCE_SEED
MIN_LEARNED_WEIGHT = store_module.MIN_LEARNED_WEIGHT

DIM = 8


def vector(*values: float) -> np.ndarray:
    """A unit-length vector -- the shape Ollama returns."""
    raw = np.zeros(DIM, dtype=np.float32)
    raw[: len(values)] = values
    return raw / np.linalg.norm(raw)


LIGHT_ON = vector(1, 0, 0)
LIGHT_ON_SIMILAR = vector(1, 0.05, 0)
COVER = vector(0, 1, 0)

ACTION_LIGHT = {
    "type": "actions",
    "actions": [
        {"domain": "light", "service": "turn_on", "data": {"entity_id": "light.lr"}}
    ],
}
ACTION_COVER = {
    "type": "actions",
    "actions": [{"domain": "cover", "service": "open_cover", "data": {}}],
}


@pytest.fixture
def store(tmp_path: pathlib.Path):
    s = Store(str(tmp_path / "test.db"))
    s.setup()
    yield s
    s.close()


def card(store, sentence: str, vec: np.ndarray, action: dict, source=SOURCE_SEED):
    """A card that may answer straight away, unless a test says otherwise.

    Seeded cards are exempt from the confirmation rule -- they come from Home
    Assistant's own templates rather than from watching the fallback agent.
    """
    return store.add_example(
        utterance=sentence,
        norm=sentence.lower(),
        language="en",
        action=action,
        source=source,
        embedding=vec,
    )


# -- Basics -------------------------------------------------------------------


def test_empty_box_finds_nothing(store) -> None:
    assert store.search(LIGHT_ON) is None


def test_card_is_found_again(store) -> None:
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT)

    hit = store.search(LIGHT_ON)

    assert hit is not None
    assert hit.action == ACTION_LIGHT
    assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_similar_sentence_finds_the_same_card(store) -> None:
    """The whole point: not just exact matches."""
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT)

    hit = store.search(LIGHT_ON_SIMILAR)

    assert hit is not None
    assert hit.action == ACTION_LIGHT
    assert 0.99 < hit.score < 1.0


def test_nearest_card_wins(store) -> None:
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT)
    card(store, "Open the blinds", COVER, ACTION_COVER)

    assert store.search(COVER).action == ACTION_COVER
    assert store.search(LIGHT_ON).action == ACTION_LIGHT


# -- Duplicates ---------------------------------------------------------------


def test_identical_card_is_not_stored_twice(store) -> None:
    first = card(store, "Light on", LIGHT_ON, ACTION_LIGHT)
    second = card(store, "Light on", LIGHT_ON, ACTION_LIGHT)

    assert first == second
    assert store.stats()["examples"] == 1


def test_frequent_sentence_weighs_more(store) -> None:
    example_id = card(store, "Light on", LIGHT_ON, ACTION_LIGHT)
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT)

    assert store.get_example(example_id)["weight"] == pytest.approx(2.0)


def test_same_sentence_other_action_is_a_new_card(store) -> None:
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT)
    card(store, "Light on", COVER, ACTION_COVER)

    assert store.stats()["examples"] == 2


# -- Restart ------------------------------------------------------------------


def test_index_survives_a_restart(tmp_path: pathlib.Path) -> None:
    path = str(tmp_path / "restart.db")

    first = Store(path)
    first.setup()
    card(first, "Light on", LIGHT_ON, ACTION_LIGHT)
    first.close()

    second = Store(path)
    second.setup()
    try:
        assert second.stats()["indexed"] == 1
        assert second.search(LIGHT_ON).action == ACTION_LIGHT
    finally:
        second.close()


# -- Robustness ---------------------------------------------------------------


def test_a_different_embedding_model_yields_no_hit(store) -> None:
    """If the user switches models, the stored vectors no longer line up.

    Better to find nothing than something wrong -- the request goes to the LLM.
    """
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT)

    assert store.search(np.ones(DIM * 2, dtype=np.float32) / np.sqrt(DIM * 2)) is None


def test_card_without_a_vector_stays_out_of_the_index(store) -> None:
    """Can happen when Ollama was down at the moment of learning."""
    store.add_example(
        utterance="Light on",
        norm="light on",
        language="en",
        action=ACTION_LIGHT,
        source=SOURCE_SEED,
        embedding=None,
    )

    assert store.stats() == {"examples": 1, "decisions": 0, "indexed": 0}
    assert len(store.examples_without_embedding()) == 1


def test_a_missing_vector_can_be_filled_in_later(store) -> None:
    example_id = store.add_example(
        utterance="Light on",
        norm="light on",
        language="en",
        action=ACTION_LIGHT,
        source=SOURCE_SEED,
    )

    store.set_embedding(example_id, LIGHT_ON)

    assert store.stats()["indexed"] == 1
    assert store.search(LIGHT_ON).example_id == example_id


def test_another_language_is_discarded(store) -> None:
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT)

    assert store.search(LIGHT_ON, language="en") is not None
    assert store.search(LIGHT_ON, language="de") is None


# -- Log ----------------------------------------------------------------------


def test_decisions_are_logged(store) -> None:
    decision_id = store.log_decision(
        utterance="Light on",
        language="en",
        mode="observe",
        tier="abstain",
        latency_ms=12,
    )
    store.attach_observation(decision_id, ACTION_LIGHT)

    assert store.stats()["decisions"] == 1


def test_hits_are_counted(store) -> None:
    example_id = card(store, "Light on", LIGHT_ON, ACTION_LIGHT)

    store.mark_hit(example_id)
    store.mark_hit(example_id)

    row = store.get_example(example_id)
    assert row["hits"] == 2
    assert row["last_hit_at"] is not None


def test_using_the_store_without_setup_complains(tmp_path: pathlib.Path) -> None:
    with pytest.raises(RuntimeError):
        Store(str(tmp_path / "x.db")).stats()


def test_key_order_does_not_create_a_second_card(store) -> None:
    """Deduplication hinges on a stable rendering of the action.

    The same action arriving with its keys in a different order must land on
    the existing card, not create a near-duplicate that splits the weight.
    """
    action_a = {"type": "actions", "actions": [{"domain": "light", "service": "on"}]}
    action_b = {"actions": [{"service": "on", "domain": "light"}], "type": "actions"}

    first = card(store, "Light on", LIGHT_ON, action_a)
    second = card(store, "Light on", LIGHT_ON, action_b)

    assert first == second
    assert store.stats()["examples"] == 1


def spread_vector(index: int) -> np.ndarray:
    """A distinct unit vector per index, so hits can be told apart."""
    raw = np.zeros(DIM, dtype=np.float32)
    raw[index % DIM] = 1.0
    raw[(index * 7) % DIM] += 0.5
    return raw / np.linalg.norm(raw)


def test_index_stays_correct_while_it_grows(store) -> None:
    """The index buffers grow by doubling rather than by copying every append.

    Growing is where an off-by-one silently loses or duplicates rows, so this
    walks well past the initial capacity and checks the bookkeeping holds.
    """
    total = 700  # past the 256 starting capacity and its first doubling

    for i in range(total):
        store.add_example(
            utterance=f"sentence {i}",
            norm=f"sentence {i}",
            language="en",
            action={"type": "actions", "actions": [{"domain": "light", "i": i}]},
            source=SOURCE_SEED,
            embedding=spread_vector(i),
        )

    stats = store.stats()
    assert stats["examples"] == total
    assert stats["indexed"] == total


def test_the_right_card_is_found_after_growth(store) -> None:
    """A grown buffer must not shuffle ids and vectors apart."""
    wanted = np.zeros(DIM, dtype=np.float32)
    wanted[3] = 1.0

    for i in range(400):
        store.add_example(
            utterance=f"filler {i}",
            norm=f"filler {i}",
            language="en",
            action={"filler": i},
            source=SOURCE_SEED,
            embedding=spread_vector(i + 1),
        )
    target = store.add_example(
        utterance="the one",
        norm="the one",
        language="en",
        action=ACTION_LIGHT,
        source=SOURCE_SEED,
        embedding=wanted,
    )

    hit = store.search(wanted)

    assert hit is not None
    assert hit.example_id == target
    assert hit.utterance == "the one"
    assert hit.action == ACTION_LIGHT


# -- Confirmation before a learned card may answer ----------------------------


def test_a_learned_card_does_not_answer_the_first_time(store) -> None:
    """This is what replaced guessing whether a follow-up was a correction.

    If the fallback agent gets something wrong once, the card it leaves behind
    is never used. Being wrong twice for the same sentence does not happen in
    practice, so no heuristic is needed to spot the mistake.
    """
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_LEARNED)

    assert store.search(LIGHT_ON) is None


def test_saying_it_again_confirms_the_card(store) -> None:
    """What the user really says regularly earns its way in on the second time."""
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_LEARNED)
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_LEARNED)

    hit = store.search(LIGHT_ON)

    assert hit is not None
    assert hit.action == ACTION_LIGHT


def test_confirmation_takes_effect_without_a_restart(store) -> None:
    """The promotion happens in the live index, not just in the database."""
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_LEARNED)
    assert store.search(LIGHT_ON) is None

    card(store, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_LEARNED)

    assert store.search(LIGHT_ON) is not None


def test_confirmation_survives_a_restart(tmp_path: pathlib.Path) -> None:
    path = str(tmp_path / "confirm.db")

    first = Store(path)
    first.setup()
    card(first, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_LEARNED)
    card(first, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_LEARNED)
    first.close()

    second = Store(path)
    second.setup()
    try:
        assert second.search(LIGHT_ON) is not None
    finally:
        second.close()


def test_seeded_cards_answer_immediately(store) -> None:
    """They come from curated templates, not from watching an agent guess."""
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_SEED)

    assert store.search(LIGHT_ON) is not None


def test_an_unconfirmed_card_does_not_hide_a_confirmed_one(store) -> None:
    """Masking, not filtering after the fact.

    A near-perfect unconfirmed match must not shadow the slightly more distant
    card that is actually allowed to answer.
    """
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_SEED)
    card(store, "Light on please", LIGHT_ON_SIMILAR, ACTION_COVER, SOURCE_LEARNED)

    hit = store.search(LIGHT_ON_SIMILAR)

    assert hit is not None
    assert hit.action == ACTION_LIGHT


def test_cards_awaiting_confirmation_are_counted(store) -> None:
    """The diagnostic sensor shows how many are still warming up."""
    card(store, "Light on", LIGHT_ON, ACTION_LIGHT, source=SOURCE_SEED)
    card(store, "Blinds up", COVER, ACTION_COVER, source=SOURCE_LEARNED)

    stats = store.card_stats()

    assert stats["total"] == 2
    assert stats["awaiting_confirmation"] == 1


def test_reindexing_a_card_does_not_duplicate_it(store) -> None:
    """Setting a vector twice must replace it, not add a second row.

    An orphaned row would stay in the index with a stale vector and could keep
    winning matches for a card whose content has moved on.
    """
    example_id = store.add_example(
        utterance="Light on",
        norm="light on",
        language="en",
        action=ACTION_LIGHT,
        source=SOURCE_SEED,
    )

    store.set_embedding(example_id, LIGHT_ON)
    store.set_embedding(example_id, COVER)

    assert store.stats()["indexed"] == 1
    # The second vector is the one that counts.
    assert store.search(COVER).example_id == example_id
    assert store.search(LIGHT_ON) is None or store.search(LIGHT_ON).score < 0.5
