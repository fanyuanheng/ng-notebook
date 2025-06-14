import os
import pytest
import pandas as pd
from pathlib import Path
from ng_notebook.services.sqlite_store import SQLiteStore

@pytest.fixture
def test_db_dir(tmp_path):
    """Create a temporary directory for test database."""
    db_dir = tmp_path / "test_sqlite_db"
    db_dir.mkdir(exist_ok=True)
    return db_dir

@pytest.fixture
def sqlite_store(test_db_dir):
    """Create a SQLiteStore instance with test database directory."""
    # Set environment variable for SQLite DB directory
    os.environ["SQLITE_DB_DIR"] = str(test_db_dir)
    # Create store instance
    store = SQLiteStore()
    yield store
    # Cleanup
    if os.path.exists(store.db_path):
        os.remove(store.db_path)

def test_init_db(sqlite_store):
    """Test database initialization."""
    with sqlite_store.get_connection() as conn:
        cursor = conn.cursor()
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "uploaded_files" in tables
        assert "excel_sheets" in tables

def test_add_csv_file(sqlite_store, tmp_path):
    """Test adding a CSV file to the store."""
    # Create a test CSV file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    df.to_csv(csv_path, index=False)
    
    # Add file to store
    result = sqlite_store.add_csv_file(str(csv_path))
    
    # Verify result
    assert result["filename"] == "test.csv"
    assert result["row_count"] == 3
    assert result["column_count"] == 2
    
    # Verify data in database
    with sqlite_store.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE filename = ?", ("test.csv",))
        assert cursor.fetchone()[0] == 1
        
        # Get table name
        cursor.execute("SELECT id FROM uploaded_files WHERE filename = ?", ("test.csv",))
        file_id = cursor.fetchone()[0]
        table_name = f"csv_{file_id}_test_csv"  # Fixed table name to match the actual implementation
        
        # Check data in table
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        assert cursor.fetchone()[0] == 3

def test_get_file_metadata(sqlite_store, tmp_path):
    """Test retrieving file metadata."""
    # Add a test file
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"col1": [1, 2, 3]})
    df.to_csv(csv_path, index=False)
    result = sqlite_store.add_csv_file(str(csv_path))
    
    # Get metadata
    metadata = sqlite_store.get_file_metadata(result["file_id"])
    
    # Verify metadata
    assert metadata["filename"] == "test.csv"
    assert metadata["file_type"] == "csv"
    assert len(metadata["sheets"]) == 1
    assert metadata["sheets"][0]["row_count"] == 3 