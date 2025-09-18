# src/app/tasks.py
import logging, queue, threading, time
from typing import Any, Dict, Optional, Callable

log = logging.getLogger(__name__)
_q: "queue.Queue[tuple[int, Dict[str, Any]]]" = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_handler: Optional[Callable[[int, Dict[str, Any]], None]] = None

def _worker():
    while True:
        article_id, payload = _q.get()
        if article_id is None:  # poison pill
            break
        try:
            if _handler:
                _handler(article_id, payload)
            else:
                log.info("[tasks] summarize(%s): %s", article_id, payload)
        except Exception as e:
            log.exception("task failed for %s: %s", article_id, e)
        finally:
            _q.task_done()

def start_worker(handler: Callable[[int, Dict[str, Any]], None] = None):
    global _worker_thread, _handler
    if _worker_thread: return
    _handler = handler
    _worker_thread = threading.Thread(target=_worker, daemon=True)
    _worker_thread.start()

def stop_worker():
    if _worker_thread:
        _q.put((None, {}))  # poison pill
        _worker_thread.join(timeout=2)

def enqueue_summarize(article_id: int, payload: Dict[str, Any] | None = None):
    _q.put((article_id, payload or {}))
