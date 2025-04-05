# cupcake_journal.py
from datetime import datetime

class CupcakeJournal:
    def __init__(self, journal_path="journal.txt"):
        self.journal_path = journal_path

    def log_entry(self, emotion, category, content, theme="desconhecido", tag=None):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        tag_line = f"🏷️ Tag: {tag}\n" if tag else ""
        entry = (
            f"[{timestamp}] 🌿 {category.upper()} ({emotion})\n"
            f"🗒️ Tema: {theme}\n"
            f"{tag_line}"
            f"💭 {content}\n"
            f"{'-'*40}\n"
        )
        with open(self.journal_path, "a", encoding="utf-8", errors="replace") as file:
            file.write(entry)
        print(f"📘 Entrada registrada no diário: {category} ({emotion})")

