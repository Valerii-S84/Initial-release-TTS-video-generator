from __future__ import annotations

import os
import importlib


def test_db_replica_session_sqlite(tmp_path):
    # Set primary and replica to temp SQLite files
    dbfile = tmp_path / "p.db"
    rfile = tmp_path / "r.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{dbfile.as_posix()}"
    os.environ["DATABASE_REPLICA_URL"] = f"sqlite:///{rfile.as_posix()}"
    import backend.db as db
    importlib.reload(db)

    # Replica session should be distinct and usable
    gen = db.get_replica_db()
    sess = next(gen)
    try:
        conn = sess.connection()
        assert conn is not None
    finally:
        try:
            gen.close()  # type: ignore[attr-defined]
        except Exception:
            pass

