from database import Base, engine
from models import ArtistDB, ReleaseDB, QuarterlyReportDB

# Создание всех таблиц
Base.metadata.create_all(bind=engine)