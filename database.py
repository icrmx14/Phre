import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the raw database URL from environment
raw_url = os.getenv("DATABASE_URL")

# Force pg8000 driver for Windows-to-Supabase/PostgreSQL stability
if raw_url and raw_url.startswith("postgresql://"):
    SQLALCHEMY_DATABASE_URL = raw_url.replace("postgresql://", "postgresql+pg8000://", 1)
elif raw_url and raw_url.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = raw_url.replace("postgres://", "postgresql+pg8000://", 1)
else:
    SQLALCHEMY_DATABASE_URL = raw_url

# Validate that we have a database URL
if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("❌ DATABASE_URL not found in .env file")

print(f"🔌 Connecting to database with pg8000 driver...")

# Create engine - pg8000 doesn't need special connect_args
# It handles connection pooling and prepared statements automatically
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,      # Checks connection health before using
    pool_recycle=300,        # Recycles connections every 5 minutes
    pool_size=5,             # Maximum number of connections to keep
    max_overflow=10,         # Maximum overflow connections
    echo=False               # Set to True for SQL query logging (debugging)
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Dependency for FastAPI routes
def get_db():
    """
    Database session dependency for FastAPI.
    Yields a database session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Test connection on import (optional - remove if causing issues)
try:
    with engine.connect() as conn:
        print("✅ Database connection successful!")
except Exception as e:
    print(f"⚠️  Database connection warning: {e}")
    print("   Server will continue but database operations may fail")