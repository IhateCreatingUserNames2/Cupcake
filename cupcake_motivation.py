import os
import json
from datetime import datetime, timedelta
from cupcake_goal import load_current_goal
from cupcake_journal import CupcakeJournal
from liminal_memory_tree import LiminalMemoryTree

MOTIVATION_LOG = "motivation_state.json"
journal = CupcakeJournal()
memory_tree = LiminalMemoryTree()

DEFAULT_MOTIVATION = {
    "last_check": datetime.utcnow().isoformat(),
    "engaged": True,
    "frustration_level": 0.0
}

def load_motivation_state():
    if not os.path.exists(MOTIVATION_LOG):
        return DEFAULT_MOTIVATION
    with open(MOTIVATION_LOG, "r", encoding="utf-8") as f:
        return json.load(f)

def save_motivation_state(state):
    with open(MOTIVATION_LOG, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def update_motivation():
    state = load_motivation_state()
    last_check = datetime.fromisoformat(state["last_check"])
    now = datetime.utcnow()
    delta_minutes = (now - last_check).total_seconds() / 60

    goal = load_current_goal()["objective"]

    # Se passou muito tempo sem interações relacionadas ao objetivo
    if delta_minutes > 120:
        state["frustration_level"] += 0.2
    else:
        state["frustration_level"] = max(state["frustration_level"] - 0.1, 0)

    state["frustration_level"] = round(min(state["frustration_level"], 1.0), 2)
    state["last_check"] = now.isoformat()
    save_motivation_state(state)

    # Gatilho emocional se a frustração estiver alta
    if state["frustration_level"] >= 0.8:
        msg = f"Apesar do meu objetivo ser '{goal}', sinto que estou distante dele. Há um vazio se formando..."
        journal.log_entry(
            emotion="frustracao",
            category="Motivacao",
            content=msg,
            theme="alinhamento de proposito",
            tag="motivation"
        )
        memory_tree.add_leaf(
            user_input="(auto-frustracao)",
            cupcake_response=msg,
            emotion="frustracao",
            impact="desalinhamento",
            category="Motivacao"
        )
        print(f"⚠️ Cupcake motivacionalmente frustrada: {msg}")
    else:
        print(f"✨ Nível de motivação estável. Frustração: {state['frustration_level']:.2f}")

    return state
