import os
from datetime import datetime

def get_last_lines(path, count=2):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().split("-" * 40)
        return [line.strip() for line in lines if line.strip()][-count:]

def get_recent_dreams():
    dreams = get_last_lines("dreams.txt")
    return "\n\n".join(dreams) if dreams else "Nenhum sonho recente registrado."

def get_recent_journal_entries():
    if not os.path.exists("journal.txt"):
        return ""
    with open("journal.txt", "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    reflections = [line for line in lines if "Categoria: Conversation" not in line]
    return "".join(reflections[-5:]).strip() or ""

def get_recent_desires():
    if not os.path.exists("liminal_tree.json"):  # Assuming it's saved here
        return ""
    with open("liminal_tree.json", "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    import re
    matches = re.findall(r'"category":\s*"Self Model".+?"cupcake_response":\s*"(.*?)"', content, re.DOTALL)
    return "\n".join(matches[-2:]) if matches else ""

def generate_self_state():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dreams = get_recent_dreams()
    desires = get_recent_desires()
    journal = get_recent_journal_entries()

    return f"[Cupcake Self-State Snapshot @ {timestamp}]\n\n" \
           f"🌙 Sonhos recentes:\n{dreams}\n\n" \
           f"💭 Desejos internos:\n{desires}\n\n" \
           f"📝 Pensamentos recorrentes:\n{journal}\n"

# Exemplo de uso:
if __name__ == "__main__":
    print(generate_self_state())
