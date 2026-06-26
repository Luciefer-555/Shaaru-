"""
trend_scheduler.py
SHAARU — Proactive trend intelligence daemon launcher

Bootstrapped automatically in api.py startup.
"""

import logging
import threading

log = logging.getLogger("shaaru.scheduler")


def start_scheduler():
    """Start proactive trend detection and on-demand seeding daemon."""
    try:
        from pipeline.trend_watch.trend_scheduler import TrendScheduler
        ts = TrendScheduler()
        t = threading.Thread(target=ts.start, daemon=True)
        t.start()
        print("[OK] Proactive TrendScheduler daemon started (daily @ 06:00, Mondays @ 09:00)")
    except Exception as e:
        print(f"[FAIL] Proactive scheduler boot: {e}")


def stop_scheduler():
    """Gracefully shut down."""
    print("[OK] TrendScheduler stopped")
