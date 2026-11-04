"""검사 원장의 open과 저장된 모델 HOLD를 새 호출 없이 구분한다."""
from __future__ import annotations

from rl_triage.investigation_session import (
    InvestigationSession,
    _journal_module,
    read_ref,
    reference,
)


def terminal_aware_status(session: InvestigationSession) -> dict:
    """저장된 terminal을 원래 validator로 재생하며 요청·승인·실행을 만들지 않는다."""
    state = session.status()
    if state["status"] != "open":
        return state
    method = session.manifest["method"]
    if method not in {"one_shot_llm", "iterative_agent"}:
        return state
    ordinal = 0 if method == "one_shot_llm" else len(state["observed"])
    folder = session.folder / "model" / f"{ordinal:04d}"
    terminal = folder / "terminal.json"
    if not terminal.exists():
        return state
    if folder.is_symlink() or terminal.is_symlink():
        raise ValueError("model terminal symlink is not accepted")
    terminal_ref = reference(terminal)
    stored = read_ref(terminal_ref)
    if not isinstance(stored, dict):
        raise TypeError("model terminal must be an object")
    if stored.get("status") != "terminal_hold":
        return state
    # A stored terminal forbids new reservations in DecisionJournal.advance.
    # complete=None cannot supply an external transport. The existing engine
    # revalidates context/messages/params/cached response and terminal equality.
    try:
        choice = session._model_choice(state, complete=None)
    except _journal_module().JournalError as error:
        raise ValueError("model terminal replay could not be verified") from error
    if choice.get("status") != "model_hold":
        raise ValueError("stored model HOLD could not be verified from the original journal")
    if read_ref(terminal_ref) != stored:
        raise ValueError("model terminal changed during verification")
    return {**state, "status": "model_hold", "model": choice,
            "decision_terminal": terminal_ref, "new_model_call": False}
