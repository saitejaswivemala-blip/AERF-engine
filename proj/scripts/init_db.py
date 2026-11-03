"""
init_db.py

Run this once (or any time you want to recreate a fresh database) to
create all tables from db/schema.sql.

Usage (from the aerf-engine/ project root):
    python scripts/init_db.py
"""

import sys
from pathlib import Path

# Allow running this script directly (python scripts/init_db.py) by
# ensuring the project root is on sys.path, since it's not a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.repository import Repository  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    repo = Repository()
    logger.info("Initializing database at %s", repo.db_path)
    repo.initialize_database()
    logger.info("Done. Tables created (or already existed).")


if __name__ == "__main__":
    main()
