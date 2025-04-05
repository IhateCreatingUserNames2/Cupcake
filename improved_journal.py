# improved_journal.py
from datetime import datetime
import os
import threading
import queue


class BufferedJournal:
    """
    An improved journal implementation with buffering and time-based file segmentation
    """

    def __init__(self, base_path="journal", segment_by="day"):
        """
        Initialize buffered journal

        Args:
            base_path: Base directory for journal files
            segment_by: How to segment files - "day", "hour", or "none"
        """
        self.base_path = base_path
        self.segment_by = segment_by
        self.buffer = queue.Queue()
        self.buffer_size = 10  # Number of entries to buffer before writing
        self.lock = threading.Lock()

        # Create directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)

        # Start background thread for writing
        self.write_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.write_thread.start()

    def _get_current_file(self):
        """Get the appropriate journal file based on segmentation strategy"""
        now = datetime.utcnow()

        if self.segment_by == "day":
            date_str = now.strftime("%Y-%m-%d")
            return os.path.join(self.base_path, f"journal_{date_str}.txt")
        elif self.segment_by == "hour":
            datetime_str = now.strftime("%Y-%m-%d_%H")
            return os.path.join(self.base_path, f"journal_{datetime_str}.txt")
        else:
            return os.path.join(self.base_path, "journal.txt")

    def log_entry(self, emotion, category, content, theme="desconhecido", tag=None):
        """Queue a journal entry to be written asynchronously"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        tag_line = f"🏷️ Tag: {tag}\n" if tag else ""
        entry = (
            f"[{timestamp}] 🌿 {category.upper()} ({emotion})\n"
            f"🗒️ Tema: {theme}\n"
            f"{tag_line}"
            f"💭 {content}\n"
            f"{'-' * 40}\n"
        )

        # Add to buffer queue
        self.buffer.put(entry)

        # Optionally print notification (can be disabled for performance)
        print(f"📘 Entrada enfileirada no diário: {category} ({emotion})")

        # If buffer is large enough, trigger a write
        if self.buffer.qsize() >= self.buffer_size:
            self._trigger_write()

    def _trigger_write(self):
        """Signal the background thread to write entries"""
        # In this implementation, the background thread checks the queue continuously
        # so no explicit trigger is needed
        pass

    def _background_writer(self):
        """Background thread for writing journal entries"""
        while True:
            # Get the current entries
            entries = []
            try:
                # Non-blocking check if there are any entries
                while not self.buffer.empty() and len(entries) < self.buffer_size:
                    entries.append(self.buffer.get_nowait())
            except queue.Empty:
                pass

            # If we have entries, write them
            if entries:
                journal_file = self._get_current_file()
                with self.lock:
                    with open(journal_file, "a", encoding="utf-8", errors="replace") as file:
                        for entry in entries:
                            file.write(entry)

                # Mark entries as done
                for _ in range(len(entries)):
                    self.buffer.task_done()

            # Sleep briefly before checking again
            time.sleep(0.1)

    def flush(self):
        """Force write all buffered entries"""
        self._trigger_write()
        self.buffer.join()  # Wait for all entries to be processed