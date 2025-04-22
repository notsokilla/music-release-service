from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from urllib.parse import quote_plus

# Получаем данные подключения
DB_USER = os.getenv('DB_USER', 'moza_postgres_user')
DB_PASSWORD = quote_plus(os.getenv('DB_PASSWORD', 'TmoI4j0nIsErOmxQdw5dRG0q8UwuZDRm'))
DB_HOST = os.getenv('DB_HOST', 'dpg-d037on9r0fns73fsbnb0-a.oregon-postgres.render.com')
DB_NAME = os.getenv('DB_NAME', 'moza_postgres')

# Формируем строку подключения с SSL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}?sslmode=require"

# Настройка движка с таймаутами
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={
        "connect_timeout": 5,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()