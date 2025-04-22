# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Используем точные данные из Render
DATABASE_URL = "postgresql://moza_postgres_user:TmoI4j0nIsErOmxQdw5dRG0q8UwuZDRm@dpg-d037on9r0fns73fsbnb0-a:5432/moza_postgres"

# Альтернативно можно использовать переменную окружения (рекомендуется)
# DATABASE_URL = os.environ.get('DATABASE_URL')

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()