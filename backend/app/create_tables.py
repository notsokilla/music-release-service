# create_tables.py
from sqlalchemy import text
from database import Base, engine
from backend.app.models import ArtistDB, SplitDB

def reset_database():
    # Удаляем все таблицы
    Base.metadata.drop_all(bind=engine)
    # Создаем заново
    Base.metadata.create_all(bind=engine)
    print("Database reset complete!")

if __name__ == "__main__":
    reset_database()