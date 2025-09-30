import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import datetime

logging.basicConfig(level=logging.INFO)

# --- Database Connection ---
# Connection details will be provided by environment variables set in docker-compose.
DB_USER = os.environ.get("POSTGRES_USER", "postgres")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.environ.get("POSTGRES_HOST", "timescaledb")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "postgres")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Data Models (SQLAlchemy ORM) ---

class InterventionRecord(Base):
    """
    Represents a record of a single intervention proposed by the system.
    This table stores the 'what' and 'why' of a change.
    """
    __tablename__ = "intervention_records"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    bottleneck_identifier = Column(String, index=True)
    root_cause = Column(String)
    confidence = Column(Float)
    proposal_description = Column(Text)
    original_code = Column(Text)
    optimized_code = Column(Text)
    # This would later be updated with the result of the PR (e.g., 'MERGED', 'REJECTED')
    status = Column(String, default="PROPOSED")

class HistoricalPerformanceData(Base):
    """
    Represents a snapshot of system performance metrics over time.
    This is the data the CIE would use to make its decisions.
    NOTE: In a real system, TimescaleDB's hypertable feature would be used here.
    """
    __tablename__ = "historical_performance_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    service_name = Column(String, index=True)
    metric_name = Column(String, index=True)
    metric_value = Column(Float)

# --- Meta-Learning Engine ---

class MetaLearningEngine:
    """
    Handles the storage and retrieval of historical data to enable long-term learning.
    """
    def __init__(self):
        self.db_session = SessionLocal()

    def create_database_tables(self):
        """
        Creates the necessary database tables if they don't already exist.
        This should be called on service startup.
        """
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Meta-Learning Engine: Database tables checked/created successfully.")
        except SQLAlchemyError as e:
            logging.error(f"Meta-Learning Engine: Error creating database tables: {e}")

    def record_intervention(self, data: dict):
        """
        Saves a record of a proposed intervention to the database.
        """
        try:
            record = InterventionRecord(
                bottleneck_identifier=data.get("bottleneck_identifier"),
                root_cause=data.get("root_cause"),
                confidence=data.get("confidence"),
                proposal_description=data.get("proposal_description"),
                original_code=data.get("original_code"),
                optimized_code=data.get("optimized_code"),
            )
            self.db_session.add(record)
            self.db_session.commit()
            logging.info(f"Meta-Learning Engine: Successfully recorded intervention for '{data.get('bottleneck_identifier')}'.")
            return record
        except SQLAlchemyError as e:
            logging.error(f"Meta-Learning Engine: Error recording intervention: {e}")
            self.db_session.rollback()
            return None

# Global instance for easy import
MLE = MetaLearningEngine()
