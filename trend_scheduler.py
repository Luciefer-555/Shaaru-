"""
trend_scheduler.py
SHAARU — Proactive trend intelligence daemon launcher

Bootstrapped automatically in api.py startup.
"""

import logging
import threading

log = logging.getLogger("shaaru.scheduler")

_scheduler_thread = None
_scheduler_lock = threading.Lock()
_stop_event = threading.Event()


def start_scheduler():
    """Start proactive trend detection and on-demand seeding daemon."""
    global _scheduler_thread
    with _scheduler_lock:
        if _scheduler_thread is not None and _scheduler_thread.is_alive():
            log.info("[SCHEDULER] TrendScheduler thread is already running. Skipping duplicate init.")
            return

        try:
            _stop_event.clear()
            from pipeline.trend_watch.trend_scheduler import TrendScheduler
            ts = TrendScheduler()
            
            _scheduler_thread = threading.Thread(
                target=ts.start,
                args=(_stop_event,),
                daemon=True,
                name="ShaaruTrendScheduler"
            )
            _scheduler_thread.start()
            log.info("[OK] Proactive TrendScheduler daemon started cleanly in background thread (daily @ 06:00, Mondays @ 09:00)")
        except Exception as e:
            log.error(f"[FAIL] Proactive scheduler boot failed: {e}", exc_info=True)


def stop_scheduler():
    """Gracefully shut down."""
    global _scheduler_thread
    with _scheduler_lock:
        _stop_event.set()
        if _scheduler_thread is not None and _scheduler_thread.is_alive():
            log.info("[SCHEDULER] Stopping TrendScheduler thread...")
        _scheduler_thread = None
        log.info("[OK] TrendScheduler stopped cleanly.")
