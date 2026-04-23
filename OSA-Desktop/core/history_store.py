"""SQLite-backed history store for video/live analytics."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analytics_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  ts_ms INTEGER NOT NULL,
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  frame_index INTEGER NOT NULL,
  total_products INTEGER NOT NULL,
  missing_products INTEGER NOT NULL,
  stock_pct REAL NOT NULL,
  void_count INTEGER NOT NULL,
  summary_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analytics_session_ts
  ON analytics_samples(session_id, ts_ms);
"""


@dataclass(frozen=True)
class Kpis:
    total_samples: int
    avg_stock_pct: float
    max_missing: int
    avg_voids: float


class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript(SCHEMA_SQL)

    def append_result(
        self,
        *,
        session_id: str,
        ts_ms: int,
        source_type: str,
        source_id: str,
        frame_index: int,
        results: Dict[str, Any],
    ) -> None:
        summary = results.get("summary") or {}
        total_products = int(summary.get("total_products_detected", 0) or 0)
        missing_products = int(summary.get("estimated_missing_products", 0) or 0)
        stock_pct = float(summary.get("overall_stock_percentage", 0.0) or 0.0)
        void_count = int(len(results.get("void_detections") or []))

        payload = json.dumps(summary, ensure_ascii=False, default=str)

        with self._connect() as con:
            con.execute(
                """
                INSERT INTO analytics_samples
                  (session_id, ts_ms, source_type, source_id, frame_index,
                   total_products, missing_products, stock_pct, void_count, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    int(ts_ms),
                    str(source_type),
                    str(source_id),
                    int(frame_index),
                    total_products,
                    missing_products,
                    stock_pct,
                    void_count,
                    payload,
                ),
            )

    def query_series(
        self,
        *,
        session_id: str,
        metric: str,
        limit: int = 1200,
        since_ts_ms: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        col = {
            "stock_pct": "stock_pct",
            "missing_products": "missing_products",
            "void_count": "void_count",
            "total_products": "total_products",
        }.get(metric)
        if not col:
            raise ValueError(f"Unknown metric: {metric}")

        query = f"""
            SELECT ts_ms, {col}
            FROM analytics_samples
            WHERE session_id = ?
        """
        params: List[object] = [session_id]
        if since_ts_ms is not None:
            query += " AND ts_ms >= ?"
            params.append(int(since_ts_ms))
        query += " ORDER BY ts_ms DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as con:
            rows = con.execute(query, tuple(params)).fetchall()

        # return ascending time
        rows.reverse()
        return [(int(ts), float(v)) for (ts, v) in rows]

    def query_kpis(self, *, session_id: str) -> Kpis:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT
                  COUNT(*) as n,
                  AVG(stock_pct) as avg_stock,
                  MAX(missing_products) as max_missing,
                  AVG(void_count) as avg_voids
                FROM analytics_samples
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        n = int(row[0] or 0)
        return Kpis(
            total_samples=n,
            avg_stock_pct=float(row[1] or 0.0),
            max_missing=int(row[2] or 0),
            avg_voids=float(row[3] or 0.0),
        )

    def list_products(self, *, session_id: str, since_ts_ms: Optional[int] = None) -> List[str]:
        query = """
            SELECT summary_json
            FROM analytics_samples
            WHERE session_id = ?
        """
        params: List[object] = [session_id]
        if since_ts_ms is not None:
            query += " AND ts_ms >= ?"
            params.append(int(since_ts_ms))
        query += " ORDER BY ts_ms DESC LIMIT 1200"

        names: set[str] = set()
        with self._connect() as con:
            rows = con.execute(query, tuple(params)).fetchall()
        for (payload,) in rows:
            try:
                summary = json.loads(payload or "{}")
                levels = summary.get("stock_levels") or {}
                names.update(str(k) for k in levels.keys())
            except Exception:
                continue
        return sorted(names)

    def query_product_series(
        self,
        *,
        session_id: str,
        product_name: str,
        limit: int = 1200,
        since_ts_ms: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        query = """
            SELECT ts_ms, summary_json
            FROM analytics_samples
            WHERE session_id = ?
        """
        params: List[object] = [session_id]
        if since_ts_ms is not None:
            query += " AND ts_ms >= ?"
            params.append(int(since_ts_ms))
        query += " ORDER BY ts_ms DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as con:
            rows = con.execute(query, tuple(params)).fetchall()

        out: List[Tuple[int, float]] = []
        for ts_ms, payload in reversed(rows):
            try:
                summary = json.loads(payload or "{}")
                levels = summary.get("stock_levels") or {}
                pdata = levels.get(product_name) or {}
                pct = float(pdata.get("stock_percentage", 0.0) or 0.0)
                out.append((int(ts_ms), pct))
            except Exception:
                out.append((int(ts_ms), 0.0))
        return out

