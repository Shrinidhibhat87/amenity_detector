"""Python file/module that handles the amenity data including storing and retrieval."""
import sqlite3
import os
import csv
import logging
import pandas as pd

from pathlib import Path
from typing import Dict
from datetime import datetime
from typing import List

class AmenityDataManager:
    """
    Manages storage and retrieval of amenity data in both CSV and SQLite formats.
    """
    def __init__(self, output_dir: str, amenity_schema: Dict[str, List[str]], logger=None):
        """
        Initialize the data manager with output directory.
        
        Args:
            output_dir: Directory to store output files
            amenity_schema: Dictionary mapping room types to lists of amenities
            logger: Optional logger for logging messages
        """
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.amenity_schema = amenity_schema

        # Get flattened list of all amenities for CSV headers
        self.all_amenities = []
        for room_type, amenities in amenity_schema.items():
            for amenity in amenities:
                self.all_amenities.append(f"{room_type}_{amenity}")

        # Initialize CSV file
        self.csv_path = self.output_dir / "amenities.csv"
        
        # Initialize SQLite database
        self.db_path = self.output_dir / "amenities.db"
        self._initialize_db()

        self.logger.info(f"Data storage initialized at {output_dir}")

    def _initialize_db(self):
        """Initialize the SQLite database with appropriate schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create images table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT UNIQUE,
            description TEXT,
            processed_at TIMESTAMP
        )
        ''')
        
        # Create amenities table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS amenities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            room_type TEXT,
            amenity_name TEXT,
            is_present BOOLEAN,
            FOREIGN KEY (image_id) REFERENCES images (id),
            UNIQUE(image_id, room_type, amenity_name)
        )
        ''')
        
        conn.commit()
        conn.close()

    def save_results(self, 
                    image_path: str, 
                    amenities: Dict[str, Dict[str, bool]],
                    description: str,
                    detected_amenities: Dict[str, bool]) -> None:
        """
        Save detection results to both CSV and SQLite formats.
        
        Args:
            image_path: Path to the processed image
            amenities: Dictionary of detected amenities
            description: Generated property description
            detected_amenities: Flat dictionary of all detected amenities
        """
        # Get image name from path
        image_name = os.path.basename(image_path)
        # Save to SQLite
        self._save_to_sqlite(image_name, image_path, amenities, description)
        
        # Save to CSV
        self._save_to_csv(image_name, image_path, detected_amenities, description)

        self.logger.info(f"Results for {image_path} saved to database and CSV")

    def _save_to_sqlite(
        self,
        image_name: str,
        image_path: str,
        amenities: Dict[str, Dict[str, bool]],
        description: str
    ) -> None:
        """Save results to SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Insert or update image record
        # ? indicates the placeholder for the values to be inserted
        cursor.execute(
            "INSERT OR REPLACE INTO images (image_path, description, processed_at) VALUES (?, ?, ?)",
            (image_path, description, datetime.now().isoformat())
        )

        # Get the image_id (either newly inserted or existing)
        cursor.execute("SELECT id FROM images WHERE image_path = ?", (image_path,))
        image_id = cursor.fetchone()[0]
        
        # Insert amenity records
        for room_type, room_amenities in amenities.items():
            for amenity_name, is_present in room_amenities.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO amenities (image_id, room_type, amenity_name, is_present) VALUES (?, ?, ?, ?)",
                    (image_id, room_type, amenity_name, is_present)
                )
        
        conn.commit()
        conn.close()

    def _save_to_csv(
        self,
        image_name: str,
        image_path: str,
        detected_amenities: Dict[str, bool],
        description: str
    ) -> None:
        """Save results to CSV file."""
        # Flatten the amenities dictionary for CSV storage
        flat_data = {
            "image_name": image_name,
            "image_path": image_path,
            "description": description
        }

        # Add all detected amenities
        for amenity_name, is_present in detected_amenities.items():
            flat_data[amenity_name] = int(is_present)

        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

        # Get current data field names
        current_fields = list(flat_data.keys())

        if file_exists:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                existing_rows = list(reader)
                existing_fieldnames = reader.fieldnames if reader.fieldnames else []

            # Detect new columns if any
            new_columns = [field for field in current_fields if field not in existing_fieldnames]

            if new_columns:
                # Add new columns to existing fieldnames
                updated_fieldnames = existing_fieldnames + new_columns

                # Rewrite the CSV with updated headers and existing data
                with open(self.csv_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=updated_fieldnames)
                    writer.writeheader()
                    for row in existing_rows:
                        # Fill the missing columns with 0
                        for col in new_columns:
                            row[col] = 0
                        writer.writerow(row)
                
                fieldnames = updated_fieldnames
            else:
                fieldnames = existing_fieldnames

        else:
            # First write: use amenity schema to get all possible amenities
            all_amenities = sorted({a for amenities in self.amenity_schema.values() for a in amenities})
            fieldnames = ["image_name", "image_path", "description"] + all_amenities

        # Ensure all amenity columns are included in the data, even if not in current image
        for field in fieldnames:
            if field not in flat_data:
                flat_data[field] = 0

        # Write to CSV
        mode = 'a' if file_exists else 'w'
        with open(self.csv_path, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_data)

    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary of all processed images and their amenities.
        
        Returns:
            DataFrame containing image paths and amenity counts
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            
            # Query to get image paths and amenity counts
            query = """
            SELECT i.image_path, i.description, 
                   SUM(CASE WHEN a.is_present = 1 THEN 1 ELSE 0 END) as amenity_count
            FROM images i
            LEFT JOIN amenities a ON i.id = a.image_id
            GROUP BY i.image_path, i.description
            ORDER BY amenity_count DESC
            """

            df = pd.read_sql_query(query, conn)
            conn.close()
            return df

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error retrieving results summary: {e}")
            return pd.DataFrame(columns=["image_name", "image_path", "description", "amenity_count"])

    def get_all_results_as_dataframe(self) -> pd.DataFrame:
        """
        Get all results as a single dataframe with each amenity as a column.
        
        Returns:
            DataFrame with each image as a row and each amenity as a column
        """
        try:
            # Read the CSV file directly as it's already in the desired format
            df = pd.read_csv(self.csv_path)
            return df
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
            
            # As a fallback, reconstruct from SQLite
            try:
                conn = sqlite3.connect(str(self.db_path))
                
                # Get all images
                images_df = pd.read_sql_query("SELECT id, image_path, description FROM images", conn)
                
                # Get all amenities
                amenities_df = pd.read_sql_query(
                    "SELECT image_id, room_type, amenity_name, is_present FROM amenities", 
                    conn
                )
                
                # Create pivot table of amenities
                if not amenities_df.empty:
                    pivot_df = amenities_df.pivot_table(
                        index='image_id', 
                        columns=['room_type', 'amenity_name'], 
                        values='is_present',
                        fill_value=0
                    )
                    
                    # Flatten column names
                    pivot_df.columns = [f"{room}_{amenity}" for room, amenity in pivot_df.columns]
                    
                    # Reset index to merge with images_df
                    pivot_df = pivot_df.reset_index()
                    
                    # Merge with images data
                    result_df = pd.merge(images_df, pivot_df, left_on='id', right_on='image_id')
                    result_df = result_df.drop(columns=['id', 'image_id'])
                else:
                    result_df = images_df.drop(columns=['id'])
                
                conn.close()
                return result_df
                
            except Exception as inner_e:
                self.logger.error(f"Error reconstructing data from SQLite: {inner_e}")
                return pd.DataFrame()
