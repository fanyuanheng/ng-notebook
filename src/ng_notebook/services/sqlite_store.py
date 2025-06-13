import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Any
import os
from pathlib import Path
import logging
import re
from contextlib import contextmanager
from ..core.config import SQLITE_DB_DIR

# Get the dedicated SQLite store logger
logger = logging.getLogger('ng_notebook.services.sqlite_store')

def sanitize_table_name(name: str) -> str:
    """
    Sanitize a string to be used as a SQLite table name.
    - Removes or replaces special characters
    - Ensures the name starts with a letter
    - Limits length to SQLite's default limit
    """
    # Replace any non-alphanumeric characters with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
    # Ensure the name starts with a letter
    if not sanitized[0].isalpha():
        sanitized = 't_' + sanitized
    # Ensure the name is not too long
    if len(sanitized) > 63:  # SQLite's default limit
        sanitized = sanitized[:63]
    return sanitized.lower()

class SQLiteStore:
    def __init__(self):
        """Initialize SQLite store with proper directory setup."""
        try:
            # Ensure the directory exists with proper permissions
            os.makedirs(SQLITE_DB_DIR, exist_ok=True)
            os.chmod(SQLITE_DB_DIR, 0o755)
            
            self.db_path = SQLITE_DB_DIR / "documents.db"
            logger.info("Initialized SQLite store with database path: %s", self.db_path)
            
            # Initialize database schema
            self._init_db()
            
        except Exception as e:
            logger.error("Failed to initialize SQLite store: %s", str(e), exc_info=True)
            raise

    def _init_db(self):
        """Initialize database schema with proper error handling."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create uploaded_files table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(filename)
                    )
                """)
                
                # Create excel_sheets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS excel_sheets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_id INTEGER,
                        sheet_name TEXT NOT NULL,
                        row_count INTEGER,
                        column_count INTEGER,
                        FOREIGN KEY (file_id) REFERENCES uploaded_files(id)
                    )
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            logger.error("Failed to initialize database schema: %s", str(e), exc_info=True)
            raise

    def get_connection(self):
        """Get a database connection with proper settings."""
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            
            # Configure connection for better performance and concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-2000")  # Use 2MB of cache
            conn.execute("PRAGMA temp_store=MEMORY")
            
            logger.debug("Database connection established with optimized settings")
            return conn
            
        except Exception as e:
            logger.error("Failed to establish database connection: %s", str(e), exc_info=True)
            raise

    def add_excel_file(self, file_path: str) -> Dict[str, Any]:
        """Add an Excel file to the database with proper transaction handling."""
        try:
            logger.info("Processing Excel file: %s", file_path)
            
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Excel file not found: {file_path}")
            
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            logger.debug("Excel file contains %d sheets", len(excel_file.sheet_names))
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Start transaction
                cursor.execute("BEGIN IMMEDIATE TRANSACTION")
                
                try:
                    # Insert or replace file record
                    cursor.execute("""
                        INSERT OR REPLACE INTO uploaded_files (filename, file_type)
                        VALUES (?, ?)
                    """, (os.path.basename(file_path), "excel"))
                    
                    file_id = cursor.lastrowid
                    logger.debug("File record created with ID: %d", file_id)
                    
                    # Process each sheet
                    for sheet_name in excel_file.sheet_names:
                        logger.debug("Processing sheet: %s", sheet_name)
                        
                        # Read sheet data
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        row_count, col_count = df.shape
                        
                        # Sanitize table name
                        safe_sheet_name = sanitize_table_name(sheet_name)
                        table_name = f"excel_{file_id}_{safe_sheet_name}"
                        logger.debug("Creating table: %s", table_name)
                        
                        # Drop existing table if it exists
                        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                        
                        # Create new table and insert data
                        df.to_sql(table_name, conn, if_exists='append', index=False)
                        
                        # Insert sheet metadata
                        cursor.execute("""
                            INSERT INTO excel_sheets (file_id, sheet_name, row_count, column_count)
                            VALUES (?, ?, ?, ?)
                        """, (file_id, sheet_name, row_count, col_count))
                        
                        logger.debug("Sheet processed: %s (%d rows, %d columns)", 
                                   sheet_name, row_count, col_count)
                    
                    # Commit transaction
                    conn.commit()
                    logger.info("Excel file processed successfully: %s", file_path)
                    
                    # Verify data
                    cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE id = ?", (file_id,))
                    file_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM excel_sheets WHERE file_id = ?", (file_id,))
                    sheet_count = cursor.fetchone()[0]
                    
                    logger.debug("Verification: %d file records, %d sheet records", 
                               file_count, sheet_count)
                    
                    return {
                        "file_id": file_id,
                        "filename": os.path.basename(file_path),
                        "sheets": excel_file.sheet_names,
                        "row_counts": {sheet: pd.read_excel(file_path, sheet).shape[0] 
                                     for sheet in excel_file.sheet_names}
                    }
                    
                except Exception as e:
                    conn.rollback()
                    logger.error("Transaction failed: %s", str(e), exc_info=True)
                    raise
                
        except Exception as e:
            logger.error("Failed to process Excel file: %s", str(e), exc_info=True)
            raise

    def add_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Add a CSV file to the database with proper transaction handling."""
        try:
            logger.info("Processing CSV file: %s", file_path)
            
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            row_count, col_count = df.shape
            logger.debug("CSV file contains %d rows and %d columns", row_count, col_count)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Start transaction
                cursor.execute("BEGIN IMMEDIATE TRANSACTION")
                
                try:
                    # Insert or replace file record
                    cursor.execute("""
                        INSERT OR REPLACE INTO uploaded_files (filename, file_type)
                        VALUES (?, ?)
                    """, (os.path.basename(file_path), "csv"))
                    
                    file_id = cursor.lastrowid
                    logger.debug("File record created with ID: %d", file_id)
                    
                    # Sanitize table name
                    safe_filename = sanitize_table_name(os.path.basename(file_path))
                    table_name = f"csv_{file_id}_{safe_filename}"
                    logger.debug("Creating table: %s", table_name)
                    
                    # Drop existing table if it exists
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    
                    # Create new table and insert data
                    df.to_sql(table_name, conn, if_exists='append', index=False)
                    
                    # Insert sheet metadata
                    cursor.execute("""
                        INSERT INTO excel_sheets (file_id, sheet_name, row_count, column_count)
                        VALUES (?, ?, ?, ?)
                    """, (file_id, "main", row_count, col_count))
                    
                    # Commit transaction
                    conn.commit()
                    logger.info("CSV file processed successfully: %s", file_path)
                    
                    # Verify data
                    cursor.execute("SELECT COUNT(*) FROM uploaded_files WHERE id = ?", (file_id,))
                    file_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM excel_sheets WHERE file_id = ?", (file_id,))
                    sheet_count = cursor.fetchone()[0]
                    
                    logger.debug("Verification: %d file records, %d sheet records", 
                               file_count, sheet_count)
                    
                    return {
                        "file_id": file_id,
                        "filename": os.path.basename(file_path),
                        "row_count": row_count,
                        "column_count": col_count
                    }
                    
                except Exception as e:
                    conn.rollback()
                    logger.error("Transaction failed: %s", str(e), exc_info=True)
                    raise
                
        except Exception as e:
            logger.error("Failed to process CSV file: %s", str(e), exc_info=True)
            raise

    def get_file_metadata(self, file_id: int) -> Dict[str, Any]:
        """Get metadata for a specific file."""
        try:
            logger.debug("Retrieving metadata for file ID: %d", file_id)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get file information
                cursor.execute("""
                    SELECT filename, file_type, upload_date
                    FROM uploaded_files
                    WHERE id = ?
                """, (file_id,))
                
                file_info = cursor.fetchone()
                if not file_info:
                    logger.warning("File not found with ID: %d", file_id)
                    return None
                
                # Get sheet information
                cursor.execute("""
                    SELECT sheet_name, row_count, column_count
                    FROM excel_sheets
                    WHERE file_id = ?
                """, (file_id,))
                
                sheets = cursor.fetchall()
                logger.debug("Found %d sheets for file ID: %d", len(sheets), file_id)
                
                return {
                    "file_id": file_id,
                    "filename": file_info[0],
                    "file_type": file_info[1],
                    "upload_date": file_info[2],
                    "sheets": [
                        {
                            "name": sheet[0],
                            "row_count": sheet[1],
                            "column_count": sheet[2]
                        }
                        for sheet in sheets
                    ]
                }
                
        except Exception as e:
            logger.error("Failed to retrieve file metadata: %s", str(e), exc_info=True)
            raise

    def get_all_files(self) -> List[Dict[str, Any]]:
        """Get metadata for all files."""
        try:
            logger.debug("Retrieving metadata for all files")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all files with their sheet counts
                cursor.execute("""
                    SELECT 
                        f.id,
                        f.filename,
                        f.file_type,
                        f.upload_date,
                        COUNT(s.id) as sheet_count
                    FROM uploaded_files f
                    LEFT JOIN excel_sheets s ON f.id = s.file_id
                    GROUP BY f.id
                    ORDER BY f.upload_date DESC
                """)
                
                files = cursor.fetchall()
                logger.debug("Found %d files", len(files))
                
                return [
                    {
                        "file_id": file[0],
                        "filename": file[1],
                        "file_type": file[2],
                        "upload_date": file[3],
                        "sheet_count": file[4]
                    }
                    for file in files
                ]
                
        except Exception as e:
            logger.error("Failed to retrieve all files metadata: %s", str(e), exc_info=True)
            raise

    def delete_file(self, file_id: int) -> bool:
        """Delete a file and its associated data."""
        try:
            logger.info("Deleting file with ID: %d", file_id)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Start transaction
                cursor.execute("BEGIN IMMEDIATE TRANSACTION")
                
                try:
                    # Get file information
                    cursor.execute("""
                        SELECT filename, file_type
                        FROM uploaded_files
                        WHERE id = ?
                    """, (file_id,))
                    
                    file_info = cursor.fetchone()
                    if not file_info:
                        logger.warning("File not found with ID: %d", file_id)
                        return False
                    
                    filename, file_type = file_info
                    
                    # Delete associated tables
                    if file_type == "excel":
                        cursor.execute("""
                            SELECT sheet_name
                            FROM excel_sheets
                            WHERE file_id = ?
                        """, (file_id,))
                        
                        sheets = cursor.fetchall()
                        for sheet in sheets:
                            table_name = f"excel_{file_id}_{sheet[0].lower().replace(' ', '_')}"
                            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                            logger.debug("Dropped table: %s", table_name)
                            
                    else:  # CSV
                        table_name = f"csv_{file_id}_{filename.lower().replace('.', '_')}"
                        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                        logger.debug("Dropped table: %s", table_name)
                    
                    # Delete sheet records
                    cursor.execute("DELETE FROM excel_sheets WHERE file_id = ?", (file_id,))
                    
                    # Delete file record
                    cursor.execute("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
                    
                    # Commit transaction
                    conn.commit()
                    logger.info("Successfully deleted file: %s", filename)
                    return True
                    
                except Exception as e:
                    conn.rollback()
                    logger.error("Transaction failed: %s", str(e), exc_info=True)
                    raise
                
        except Exception as e:
            logger.error("Failed to delete file: %s", str(e), exc_info=True)
            raise

    def get_table_data(self, file_id: int, sheet_name: Optional[str] = None, 
                      limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get data from a specific table with pagination."""
        try:
            logger.debug("Retrieving data for file ID: %d, sheet: %s", file_id, sheet_name)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get file information
                cursor.execute("""
                    SELECT filename, file_type
                    FROM uploaded_files
                    WHERE id = ?
                """, (file_id,))
                
                file_info = cursor.fetchone()
                if not file_info:
                    logger.warning("File not found with ID: %d", file_id)
                    return None
                
                filename, file_type = file_info
                
                # Determine table name
                if file_type == "excel":
                    if not sheet_name:
                        logger.warning("Sheet name required for Excel file")
                        return None
                        
                    table_name = f"excel_{file_id}_{sheet_name.lower().replace(' ', '_')}"
                else:  # CSV
                    table_name = f"csv_{file_id}_{filename.lower().replace('.', '_')}"
                
                # Get total row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_rows = cursor.fetchone()[0]
                
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Get data with pagination
                cursor.execute(f"""
                    SELECT *
                    FROM {table_name}
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                rows = cursor.fetchall()
                logger.debug("Retrieved %d rows from table: %s", len(rows), table_name)
                
                return {
                    "file_id": file_id,
                    "filename": filename,
                    "file_type": file_type,
                    "sheet_name": sheet_name,
                    "columns": [col[1] for col in columns],
                    "total_rows": total_rows,
                    "data": rows
                }
                
        except Exception as e:
            logger.error("Failed to retrieve table data: %s", str(e), exc_info=True)
            raise 