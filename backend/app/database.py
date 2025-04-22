# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from urllib.parse import quote_plus

# Получаем пароль из переменных окружения (рекомендуемый способ)
db_password = os.getenv('DB_PASSWORD', 'TmoI4j0nIsErOmxQdw5dRG0q8UwuZDRm')

# Экранируем специальные символы в пароле
escaped_password = quote_plus(db_password)

# Формируем строку подключения с SSL
DATABASE_URL = f"postgresql://moza_postgres_user:{escaped_password}@dpg-d037on9r0fns73fsbnb0-a:5432/moza_postgres?sslmode=require"

# Настройка engine с увеличенным временем ожидания
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Проверка соединения перед использованием
    pool_recycle=3600,   # Пересоздавать соединения каждый час
    connect_args={
        "connect_timeout": 10,  # 10 секунд на подключение
        "keepalives": 1,        # Включить keepalive
        "keepalives_idle": 30,  # 30 секунд бездействия
        "keepalives_interval": 10,  # Интервал проверки
        "keepalives_count": 5   # Количество попыток
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