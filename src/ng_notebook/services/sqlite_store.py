import sqlite3
import pandas as pd
from typing import List, Dict, Optional
import os
from pathlib import Path
import logging
import re
from contextlib import contextmanager
from ..core.config import SQLITE_DB_DIR

# Get logger
logger = logging.getLogger(__name__)

def sanitize_table_name(name: str) -> str:
    """Sanitize a string to be used as a SQLite table name."""
    # Replace any non-alphanumeric characters with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Ensure the name starts with a letter
    if not sanitized[0].isalpha():
        sanitized = 't_' + sanitized
    # Ensure the name is not too long
    if len(sanitized) > 63:  # SQLite's default limit
        sanitized = sanitized[:63]
    # Convert to lowercase
    sanitized = sanitized.lower()
    return sanitized

class SQLiteStore:
    def __init__(self, db_path: str = None):
        """Initialize SQLite store."""
        if db_path is None:
            db_path = str(SQLITE_DB_DIR / "excel_data.db")
        self.db_path = db_path
        logger.debug(f"Initializing SQLiteStore with db_path: {db_path}")
        self._ensure_db_directory()
        self._init_db()

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        logger.debug(f"Ensuring database directory exists: {db_dir}")
        os.makedirs(db_dir, exist_ok=True)
        logger.debug(f"Database directory exists: {os.path.exists(db_dir)}")

    @contextmanager
    def get_connection(self, timeout: float = 30.0):
        """Get a database connection with timeout."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=timeout,
                isolation_level=None  # Enable autocommit mode
            )
            conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging
            conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=-2000")  # Use 2MB of cache
            conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
            yield conn
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing database connection: {str(e)}", exc_info=True)

    def _init_db(self):
        """Initialize the database with necessary tables."""
        logger.debug("Initializing database tables")
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create uploaded_files table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create excel_sheets table to track sheets
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS excel_sheets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id INTEGER NOT NULL,
                        sheet_name TEXT NOT NULL,
                        num_rows INTEGER NOT NULL,
                        num_columns INTEGER NOT NULL,
                        FOREIGN KEY (file_id) REFERENCES uploaded_files(id)
                    )
                """)
                
                conn.commit()
                logger.debug("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise

    def add_excel_file(self, file_path: str, file_type: str) -> int:
        """Add an Excel file to the database."""
        logger.debug(f"Adding Excel file to database: {file_path}")
        try:
            # First verify the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            logger.debug(f"File exists at path: {file_path}")
            
            # Read Excel file first to verify it can be read
            logger.debug(f"Reading Excel file: {file_path}")
            excel_file = pd.ExcelFile(file_path)
            logger.debug(f"Excel file has {len(excel_file.sheet_names)} sheets")
            
            with self.get_connection() as conn:
                # Start a transaction
                conn.execute("BEGIN IMMEDIATE TRANSACTION")
                try:
                    cursor = conn.cursor()
                    
                    # Insert file record
                    filename = os.path.basename(file_path)
                    logger.debug(f"Inserting file record for: {filename}")
                    cursor.execute(
                        "INSERT OR REPLACE INTO uploaded_files (filename, file_type) VALUES (?, ?)",
                        (filename, file_type)
                    )
                    
                    # Get the file ID (handle both insert and replace cases)
                    cursor.execute("SELECT id FROM uploaded_files WHERE filename = ?", (filename,))
                    result = cursor.fetchone()
                    if not result:
                        raise Exception(f"Failed to get file_id for {filename}")
                    file_id = result[0]
                    logger.debug(f"Got file_id: {file_id} for file: {filename}")
                    
                    # Process each sheet
                    for sheet_name in excel_file.sheet_names:
                        logger.debug(f"Processing sheet: {sheet_name}")
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        
                        # Sanitize the table name
                        safe_sheet_name = sanitize_table_name(sheet_name)
                        table_name = f"excel_data_{file_id}_{safe_sheet_name}"
                        logger.debug(f"Using sanitized table name: {table_name}")
                        
                        # Drop existing table if it exists
                        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                        
                        # Create table for sheet data
                        df.to_sql(table_name, conn, if_exists='append', index=False)
                        logger.debug(f"Created table {table_name} with {len(df)} rows and {len(df.columns)} columns")
                        
                        # Record sheet metadata
                        cursor.execute("""
                            INSERT OR REPLACE INTO excel_sheets 
                            (file_id, sheet_name, num_rows, num_columns)
                            VALUES (?, ?, ?, ?)
                        """, (file_id, sheet_name, len(df), len(df.columns)))
                        logger.debug(f"Added metadata for sheet: {sheet_name}")
                    
                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"Successfully added Excel file to database: {filename}")
                    
                    # Verify the data was inserted
                    cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE filename = ?", (filename,))
                    file_count = cursor.fetchone()[0]
                    logger.debug(f"Verified file record exists: {file_count > 0}")
                    
                    cursor.execute("SELECT COUNT(*) FROM excel_sheets WHERE file_id = ?", (file_id,))
                    sheet_count = cursor.fetchone()[0]
                    logger.debug(f"Verified {sheet_count} sheets were added")
                    
                    return file_id
                    
                except Exception as e:
                    # Rollback the transaction on error
                    conn.rollback()
                    logger.error(f"Error in transaction: {str(e)}", exc_info=True)
                    raise
                
        except Exception as e:
            logger.error(f"Error adding Excel file to database: {str(e)}", exc_info=True)
            raise

    def add_csv_file(self, file_path: str, file_type: str) -> int:
        """Add a CSV file to the database."""
        logger.debug(f"Adding CSV file to database: {file_path}")
        try:
            # First verify the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            logger.debug(f"File exists at path: {file_path}")
            
            # Read CSV file first to verify it can be read
            logger.debug(f"Reading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            logger.debug(f"CSV file has {len(df)} rows and {len(df.columns)} columns")
            
            with self.get_connection() as conn:
                # Start a transaction
                conn.execute("BEGIN IMMEDIATE TRANSACTION")
                try:
                    cursor = conn.cursor()
                    
                    # Insert file record
                    filename = os.path.basename(file_path)
                    logger.debug(f"Inserting file record for: {filename}")
                    cursor.execute(
                        "INSERT OR REPLACE INTO uploaded_files (filename, file_type) VALUES (?, ?)",
                        (filename, file_type)
                    )
                    
                    # Get the file ID (handle both insert and replace cases)
                    cursor.execute("SELECT id FROM uploaded_files WHERE filename = ?", (filename,))
                    result = cursor.fetchone()
                    if not result:
                        raise Exception(f"Failed to get file_id for {filename}")
                    file_id = result[0]
                    logger.debug(f"Got file_id: {file_id} for file: {filename}")
                    
                    # Sanitize the table name
                    safe_filename = sanitize_table_name(filename)
                    table_name = f"csv_data_{file_id}_{safe_filename}"
                    logger.debug(f"Using sanitized table name: {table_name}")
                    
                    # Drop existing table if it exists
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    
                    # Create table for CSV data
                    df.to_sql(table_name, conn, if_exists='append', index=False)
                    logger.debug(f"Created table {table_name} with {len(df)} rows and {len(df.columns)} columns")
                    
                    # Record sheet metadata (CSV is treated as a single sheet)
                    cursor.execute("""
                        INSERT OR REPLACE INTO excel_sheets 
                        (file_id, sheet_name, num_rows, num_columns)
                        VALUES (?, ?, ?, ?)
                    """, (file_id, "main", len(df), len(df.columns)))
                    logger.debug("Added sheet metadata for CSV file")
                    
                    # Commit the transaction
                    conn.commit()
                    logger.debug(f"Successfully added CSV file to database: {filename}")
                    
                    # Verify the data was inserted
                    cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE filename = ?", (filename,))
                    file_count = cursor.fetchone()[0]
                    logger.debug(f"Verified file record exists: {file_count > 0}")
                    
                    cursor.execute("SELECT COUNT(*) FROM excel_sheets WHERE file_id = ?", (file_id,))
                    sheet_count = cursor.fetchone()[0]
                    logger.debug(f"Verified {sheet_count} sheets were added")
                    
                    return file_id
                    
                except Exception as e:
                    # Rollback the transaction on error
                    conn.rollback()
                    logger.error(f"Error in transaction: {str(e)}", exc_info=True)
                    raise
                
        except Exception as e:
            logger.error(f"Error adding CSV file to database: {str(e)}", exc_info=True)
            raise

    def get_file_data(self, file_id: int) -> Optional[pd.DataFrame]:
        """Get data for a specific file."""
        logger.debug(f"Getting data for file: {file_id}")
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get file info
                cursor.execute(
                    "SELECT filename, file_type FROM uploaded_files WHERE id = ?",
                    (file_id,)
                )
                file_info = cursor.fetchone()
                
                if not file_info:
                    logger.warning(f"File not found in database: {file_id}")
                    return None
                
                filename, file_type = file_info
                
                # Get sheet info
                cursor.execute(
                    "SELECT sheet_name FROM excel_sheets WHERE file_id = ?",
                    (file_id,)
                )
                sheets = cursor.fetchall()
                
                if not sheets:
                    return None
                
                # Read data from first sheet
                sheet_name = sheets[0][0]
                table_name = sanitize_table_name(f"excel_data_{file_id}_{sheet_name}")
                
                try:
                    return pd.read_sql(f"SELECT * FROM {table_name}", conn)
                except Exception as e:
                    logger.error(f"Error reading data from table {table_name}: {str(e)}", exc_info=True)
                    return None
        except Exception as e:
            logger.error(f"Error getting file data: {str(e)}", exc_info=True)
            raise

    def query_data(self, query: str) -> List[Dict]:
        """Query the SQLite database with a natural language query."""
        logger.debug(f"Querying SQLite database: {query}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE 'excel_data_%' OR name LIKE 'csv_data_%')")
            tables = cursor.fetchall()
            logger.debug(f"Found {len(tables)} tables")
            
            results = []
            for table in tables:
                table_name = table[0]
                try:
                    # Get a sample of the data
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                    rows = cursor.fetchall()
                    
                    # Get column names
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    # Format results
                    for row in rows:
                        results.append({
                            "table": table_name,
                            "data": dict(zip(columns, row))
                        })
                except sqlite3.Error as e:
                    logger.error(f"Error querying table {table_name}: {str(e)}")
                    continue
            
            logger.debug(f"Query returned {len(results)} results")
            return results

    def get_file_metadata(self, filename: str) -> Optional[Dict]:
        """Get metadata for a specific file."""
        logger.debug(f"Getting metadata for file: {filename}")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, file_type, upload_date FROM uploaded_files WHERE filename = ?",
                (filename,)
            )
            file_info = cursor.fetchone()
            
            if not file_info:
                logger.debug(f"No metadata found for file: {filename}")
                return None
            
            file_id, file_type, upload_date = file_info
            
            # Get all tables for this file
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
                (f"file_{file_id}_%",)
            )
            tables = cursor.fetchall()
            
            metadata = {
                "id": file_id,
                "filename": filename,
                "file_type": file_type,
                "upload_date": upload_date,
                "tables": []
            }
            
            for table in tables:
                table_name = table[0]
                if not table_name.endswith('_metadata'):
                    # Get row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    
                    # Get column info
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
                    
                    metadata["tables"].append({
                        "name": table_name,
                        "row_count": row_count,
                        "columns": columns
                    })
            
            logger.debug(f"Found metadata for {len(metadata['tables'])} tables")
            return metadata 