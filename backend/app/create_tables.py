from database import Base, engine
from models import ArtistDB, SplitDB
import logging

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

def create_tables():
    try:
        # Правильный порядок создания таблиц
        ArtistDB.__table__.create(bind=engine, checkfirst=True)
        SplitDB.__table__.create(bind=engine, checkfirst=True)
        print("Таблицы успешно созданы")
    except Exception as e:
        print(f"Ошибка при создании таблиц: {e}")
        raise

if __name__ == "__main__":
    create_tables()