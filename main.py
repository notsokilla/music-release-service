from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import ArtistDB, ReleaseDB, QuarterlyReportDB

app = FastAPI()

# Модели Pydantic для запросов
class Artist(BaseModel):
    name: str
    contact_info: Optional[str] = None

class Release(BaseModel):
    artist_id: int
    release_name: str
    rights_share: float
    release_date: str
    isrc: Optional[str] = None
    upc: Optional[str] = None

# Зависимость для получения сессии базы данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Функция для преобразования даты
def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if pd.isna(date_str) or not date_str or str(date_str).strip() == '':
        return None
    try:
        return datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
    except ValueError:
        return None

# Добавление артиста
@app.post("/add-artist/")
async def add_artist(artist: Artist, db: Session = Depends(get_db)):
    db_artist = ArtistDB(name=artist.name, contact_info=artist.contact_info)
    db.add(db_artist)
    db.commit()
    db.refresh(db_artist)
    return {"message": "Artist added successfully", "artist_id": db_artist.id}

# Добавление релиза
@app.post("/add-release/")
async def add_release(release: Release, db: Session = Depends(get_db)):
    db_release = ReleaseDB(
        artist_id=release.artist_id,
        release_name=release.release_name,
        rights_share=release.rights_share,
        release_date=parse_date(release.release_date),
        isrc=release.isrc,
        upc=release.upc
    )
    db.add(db_release)
    db.commit()
    db.refresh(db_release)
    return {"message": "Release added successfully", "release_id": db_release.id}

# Загрузка квартального отчета
@app.post("/upload-quarterly-report/")
async def upload_quarterly_report(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or Excel file.")

        for index, row in df.iterrows():
            db_report = QuarterlyReportDB(
                release_id=row.get('release_id'),
                total_revenue=row.get('total_revenue'),
                sales_quantity=row.get('sales_quantity'),
                royalty_amount=row.get('royalty_amount'),
                report_period=row.get('report_period')
            )
            db.add(db_report)
            db.commit()
            db.refresh(db_report)

        return {"message": "Quarterly report processed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Генерация индивидуального отчета
@app.get("/generate-report/{artist_id}")
async def generate_report(artist_id: int, db: Session = Depends(get_db)):
    artist = db.query(ArtistDB).filter(ArtistDB.id == artist_id).first()
    if not artist:
        raise HTTPException(status_code=404, detail="Artist not found")

    releases = db.query(ReleaseDB).filter(ReleaseDB.artist_id == artist_id).all()
    reports = []

    for release in releases:
        quarterly_reports = db.query(QuarterlyReportDB).filter(QuarterlyReportDB.release_id == release.id).all()
        for report in quarterly_reports:
            artist_revenue = report.total_revenue * (release.rights_share / 100)
            reports.append({
                "release_name": release.release_name,
                "report_period": report.report_period,
                "artist_share": release.rights_share,
                "artist_revenue": artist_revenue
            })

    return {"artist_name": artist.name, "reports": reports}

# Получить всех артистов (для тестирования)
@app.get("/artists/")
async def get_artists(db: Session = Depends(get_db)):
    artists = db.query(ArtistDB).all()
    return artists

# Получить все релизы (для тестирования)
@app.get("/releases/")
async def get_releases(db: Session = Depends(get_db)):
    releases = db.query(ReleaseDB).all()
    return releases

# Корневой эндпоинт
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Music Release Service"}