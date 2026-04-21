"""Basketball Reference scraper for supplementary data.

Used for:
- Wingspan data (if not in draft combine)
- Pre-1996 stats (before nba_api coverage)
- Advanced metrics calculated by BBall-Ref

Rate limiting: MAX 20 requests/minute (3+ second delays).
"""

import time
from functools import wraps


def rate_limit_bballref(min_interval=3.0):
    """Decorator to enforce BBall-Ref rate limits (20 req/min max)."""
    last_call = [0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_call[0] = time.time()
            return result

        return wrapper

    return decorator


class BballRefScraper:
    """Scraper for Basketball Reference data."""

    def __init__(self):
        pass

    @rate_limit_bballref(3.0)
    def get_player_page(self, player_slug: str):
        """Fetch and parse a player's BBall-Ref page."""
        raise NotImplementedError

    @rate_limit_bballref(3.0)
    def get_player_measurements(self, player_slug: str):
        """Extract wingspan and other measurements from player page."""
        raise NotImplementedError

    @rate_limit_bballref(3.0)
    def get_season_stats(self, player_slug: str, season: str):
        """Get stats for a specific season (useful for pre-1996)."""
        raise NotImplementedError
