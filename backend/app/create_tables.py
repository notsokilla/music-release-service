# create_tables.py
from sqlalchemy import text
from database import Base, engine
from models import ArtistDB, SplitDB
import os

def reset_database():
    # Убедимся, что URL базы данных корректный
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # Удаляем все таблицы
    Base.metadata.drop_all(bind=engine)
    # Создаем заново
    Base.metadata.create_all(bind=engine)
    print("Database reset complete!")

if __name__ == "__main__":
    reset_database()