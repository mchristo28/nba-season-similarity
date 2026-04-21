"""Caching layer using SQLite for metadata and Parquet for DataFrames."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


class CacheManager:
    """Manages caching of API responses and computed data.

    Uses SQLite for:
    - Tracking what's been fetched
    - Metadata storage
    - Request logs

    Uses Parquet for:
    - Storing DataFrames (fast columnar reads)
    """

    def __init__(self, cache_dir: Path | str = "data"):
        self.cache_dir = Path(cache_dir)
        self.raw_dir = self.cache_dir / "raw"
        self.processed_dir = self.cache_dir / "processed"
        self.features_dir = self.cache_dir / "features"
        self.db_path = self.cache_dir / "cache.db"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    entry_type TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS player_stats (
                    player_id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS team_stats (
                    team_id INTEGER,
                    season TEXT,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (team_id, season)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS player_info (
                    player_id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def has(self, key: str) -> bool:
        """Check if data exists in cache."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT 1 FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
            return result is not None

    def get(self, key: str) -> pd.DataFrame | None:
        """Retrieve data from cache."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT file_path FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
            if result is None:
                return None
            file_path = Path(result[0])
            if not file_path.exists():
                return None
            return pd.read_parquet(file_path)

    def store(self, key: str, data: pd.DataFrame, entry_type: str = "generic") -> None:
        """Store data in cache."""
        file_path = self.raw_dir / f"{key}.parquet"
        data.to_parquet(file_path, index=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache_entries (key, file_path, created_at, entry_type)
                   VALUES (?, ?, ?, ?)""",
                (key, str(file_path), datetime.now().isoformat(), entry_type),
            )

    def has_player_stats(self, player_id: int) -> bool:
        """Check if player stats are cached."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT 1 FROM player_stats WHERE player_id = ?", (player_id,)
            ).fetchone()
            return result is not None

    def get_player_stats(self, player_id: int) -> pd.DataFrame | None:
        """Get cached player stats."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT file_path FROM player_stats WHERE player_id = ?", (player_id,)
            ).fetchone()
            if result is None:
                return None
            file_path = Path(result[0])
            if not file_path.exists():
                return None
            return pd.read_parquet(file_path)

    def store_player_stats(self, player_id: int, data: pd.DataFrame) -> None:
        """Cache player stats."""
        file_path = self.raw_dir / f"player_{player_id}_stats.parquet"
        data.to_parquet(file_path, index=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO player_stats (player_id, file_path, created_at)
                   VALUES (?, ?, ?)""",
                (player_id, str(file_path), datetime.now().isoformat()),
            )

    def has_player_info(self, player_id: int) -> bool:
        """Check if player info is cached."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT 1 FROM player_info WHERE player_id = ?", (player_id,)
            ).fetchone()
            return result is not None

    def get_player_info(self, player_id: int) -> pd.DataFrame | None:
        """Get cached player info."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT file_path FROM player_info WHERE player_id = ?", (player_id,)
            ).fetchone()
            if result is None:
                return None
            file_path = Path(result[0])
            if not file_path.exists():
                return None
            return pd.read_parquet(file_path)

    def store_player_info(self, player_id: int, data: pd.DataFrame) -> None:
        """Cache player info."""
        file_path = self.raw_dir / f"player_{player_id}_info.parquet"
        data.to_parquet(file_path, index=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO player_info (player_id, file_path, created_at)
                   VALUES (?, ?, ?)""",
                (player_id, str(file_path), datetime.now().isoformat()),
            )

    def has_team_stats(self, team_id: int, season: str) -> bool:
        """Check if team stats are cached."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT 1 FROM team_stats WHERE team_id = ? AND season = ?",
                (team_id, season),
            ).fetchone()
            return result is not None

    def get_team_stats(self, team_id: int, season: str) -> pd.DataFrame | None:
        """Get cached team stats."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT file_path FROM team_stats WHERE team_id = ? AND season = ?",
                (team_id, season),
            ).fetchone()
            if result is None:
                return None
            file_path = Path(result[0])
            if not file_path.exists():
                return None
            return pd.read_parquet(file_path)

    def store_team_stats(self, team_id: int, season: str, data: pd.DataFrame) -> None:
        """Cache team stats."""
        file_path = self.raw_dir / f"team_{team_id}_{season}_stats.parquet"
        data.to_parquet(file_path, index=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO team_stats (team_id, season, file_path, created_at)
                   VALUES (?, ?, ?, ?)""",
                (team_id, season, str(file_path), datetime.now().isoformat()),
            )

    def get_all_player_stats(self) -> pd.DataFrame | None:
        """Load all cached player stats into a single DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("SELECT file_path FROM player_stats").fetchall()
        if not results:
            return None
        dfs = []
        for (file_path,) in results:
            path = Path(file_path)
            if path.exists():
                dfs.append(pd.read_parquet(path))
        return pd.concat(dfs, ignore_index=True) if dfs else None

    def get_all_player_info(self) -> pd.DataFrame | None:
        """Load all cached player info into a single DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("SELECT file_path FROM player_info").fetchall()
        if not results:
            return None
        dfs = []
        for (file_path,) in results:
            path = Path(file_path)
            if path.exists():
                dfs.append(pd.read_parquet(path))
        return pd.concat(dfs, ignore_index=True) if dfs else None
