from main import PHASE1_HYPOTHESES, PHASE2_HYPOTHESES, route_next_hypothesis
from worker import KNOWN_HYPOTHESES


def test_router_and_worker_share_one_valid_hypothesis_vocabulary():
    configured = set(PHASE1_HYPOTHESES + PHASE2_HYPOTHESES)
    assert configured <= KNOWN_HYPOTHESES


def test_router_never_repeats_completed_work():
    state = {"tried_hypotheses": list(PHASE1_HYPOTHESES)}
    assert route_next_hypothesis(state, "classification", 99) == PHASE2_HYPOTHESES[0]

    state["tried_hypotheses"] += list(PHASE2_HYPOTHESES)
    assert route_next_hypothesis(state, "classification", 100) is None
