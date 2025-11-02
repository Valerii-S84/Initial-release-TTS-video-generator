from __future__ import annotations

import os
import importlib
import pytest


@pytest.mark.isolated
def test_db_sqlite_get_db(tmp_path):
    dbfile = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{dbfile.as_posix()}"
    import backend.db as db
    importlib.reload(db)
    gen = db.get_db()
    sess = next(gen)
    try:
        conn = sess.connection()
        assert conn is not None
    finally:
        try:
            gen.close()  # type: ignore[attr-defined]
        except Exception:
            pass
