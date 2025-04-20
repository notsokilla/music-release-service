from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd
import io
import re
import tempfile
import zipfile
import os
import logging
from pydantic import BaseModel
from database import get_db, Base, engine
from models import ArtistDB, SplitDB
from fastapi.background import BackgroundTasks
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from decimal import Decimal, getcontext
from collections import defaultdict
from sqlalchemy import Column, Integer, String, Numeric
from sqlalchemy import text  # Добавьте этот импорт
from sqlalchemy import func  # Добавьте эту строку в раздел импортов

app = FastAPI(docs_url="/docs", redoc_url="/redoc")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка точности Decimal
getcontext().prec = 10

@app.get("/")
async def root():
    return {"message": "Server is running!"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модели запросов
class SplitRequest(BaseModel):
    nickname: str
    percentage: float

class TrackSplitsRequest(BaseModel):
    track_title: str
    splits: List[SplitRequest]

def get_track_name(row: pd.Series) -> str:
    """Возвращает название трека из различных возможных колонок"""
    for col in ['track_title', 'наименование', 'title', 'song_name']:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return "Unknown Track"

def convert_to_decimal(value):
    """Безопасное преобразование любого значения в Decimal"""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(float(value)))
    except (TypeError, ValueError):
        return Decimal('0')

def prepare_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим все финансовые колонки к Decimal"""
    money_cols = ['gross', 'net', 'license_fee', 'rights_holder_total', 
                 'правообладатель_вознаграждение_итого', 'комиссия_лицензиата']
    
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_to_decimal)
        else:
            df[col] = Decimal('0')
    
    return df

def parse_performers(source: str, raw_artists: str) -> List[str]:
    """Парсинг исполнителей с учетом разных форматов"""
    if pd.isna(raw_artists):
        return []
    
    raw_artists = str(raw_artists)
    
    if 'rpm' in source.lower():
        performers = []
        parts = re.split(r',\s*(?![^()]*\))', raw_artists)
        for part in parts:
            match = re.match(r'^\s*(.+?)\s*\((performer|writer|producer)\)\s*$', part)
            if match and match.group(2) == 'performer':
                performers.append(match.group(1).strip())
        return performers
    else:
        return [a.strip() for a in re.split(r',\s*(?![^/]*/)', raw_artists) if a.strip()]

async def process_report(file: UploadFile, db: Session) -> pd.DataFrame:
    """Обработка загруженного отчета"""
    contents = await file.read()
    
    if file.filename.lower().endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        df = pd.read_csv(io.BytesIO(contents))
    
    # Нормализация колонок
    df.columns = [
        col.strip().lower()
        .replace(' ', '_')
        .replace('/', '_')
        .replace('-', '_')
        for col in df.columns
    ]
    
    # Парсинг исполнителей
    source_type = 'rpm' if 'rpm' in file.filename.lower() else 'other'
    artist_column = next((col for col in df.columns if col in ['artists', 'исполнитель']), None)
    
    if artist_column:
        df['_parsed_performers'] = df[artist_column].apply(
            lambda x: parse_performers(source_type, x)
        )
    
    # Подготовка финансовых данных
    df = prepare_financial_data(df)
    return df

def calculate_profits(df: pd.DataFrame, db: Session) -> Dict[str, Dict[str, Decimal]]:
    """Единственная функция для расчета прибыли с учетом сплитов"""
    # 1. Получаем все сплиты из базы
    splits = db.query(SplitDB).all()
    track_splits = defaultdict(dict)
    for split in splits:
        track_splits[split.track_title][split.artist_nickname] = convert_to_decimal(split.split_percentage)

    df = df.apply(lambda row: apply_splits_to_profits(row, track_splits), axis=1)

    # 2. Инициализируем структуры для результатов
    results = {
        'artists': defaultdict(lambda: {'licensor': Decimal('0'), 'licensee': Decimal('0')}),
        'totals': {'licensor': Decimal('0'), 'licensee': Decimal('0')},
        'tracks': defaultdict(lambda: {'licensor': Decimal('0'), 'licensee': Decimal('0')})
    }

    for _, row in df.iterrows():
        track = row['_normalized_track_name']
        artists = row.get('_parsed_performers', []) or []
        
        # 3. Рассчитываем базовые прибыли для трека
        licensor_base = (row['gross'] - row['net'] + row['license_fee'] + row['комиссия_лицензиата'])
        licensee_base = (row['net'] + row['rights_holder_total'] + row['правообладатель_вознаграждение_итого'])

        # 4. Сохраняем данные по треку
        results['tracks'][track]['licensor'] += licensor_base
        results['tracks'][track]['licensee'] += licensee_base

        # 5. Распределяем прибыль лицензиата (платформы) поровну
        if artists:
            licensor_per_artist = licensor_base / Decimal(len(artists))
            for artist in artists:
                results['artists'][artist]['licensor'] += licensor_per_artist

        # 6. Распределяем прибыль лицензиара (артистов)
        if track in track_splits:
            # Получаем только артистов, которые есть и в треке, и в сплитах
            valid_artists = [a for a in artists if a in track_splits[track]]
            total_percent = sum(track_splits[track][a] for a in valid_artists)
            remaining_percent = max(Decimal('0'), Decimal('100') - total_percent)
            
            if valid_artists:
                # Основное распределение по сплитам
                for artist in valid_artists:
                    percent = track_splits[track][artist]
                    share = licensee_base * percent / Decimal('100')
                    results['artists'][artist]['licensee'] += share
                
                # Распределение остатка поровну между всеми артистами трека
                if remaining_percent > 0:
                    remaining_share = licensee_base * remaining_percent / Decimal('100')
                    remaining_per_artist = remaining_share / Decimal(len(artists))
                    for artist in artists:
                        results['artists'][artist]['licensee'] += remaining_per_artist
            else:
                # Если нет валидных артистов в сплитах, но есть артисты в треке
                licensee_per_artist = licensee_base / Decimal(len(artists))
                for artist in artists:
                    results['artists'][artist]['licensee'] += licensee_per_artist
        else:
            # Если нет сплитов для трека вообще
            if artists:
                licensee_per_artist = licensee_base / Decimal(len(artists))
                for artist in artists:
                    results['artists'][artist]['licensee'] += licensee_per_artist

        # 7. Суммируем общие значения
        results['totals']['licensor'] += licensor_base
        results['totals']['licensee'] += licensee_base

    return results

def generate_artist_report(artist_data: pd.DataFrame, artist: str, profits: Dict[str, Decimal]) -> bytes:
    """Генерация отчета для артиста с полными финансовыми данными"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Основные данные по трекам
        artist_data.to_excel(writer, sheet_name='Треки', index=False)
        
        # Финансовая сводка
        financial_data = pd.DataFrame({
            'Показатель': ['Прибыль от платформы', 'Прибыль от прав', 'Общая прибыль'],
            'Сумма': [
                float(profits['licensor']),
                float(profits['licensee']),
                float(profits['licensor'] + profits['licensee'])
            ]
        })
        financial_data.to_excel(writer, sheet_name='Финансы', index=False)
        
        # Детализация по трекам
        track_details = []
        for _, row in artist_data.iterrows():
            track_details.append({
                'Трек': row['_normalized_track_name'],
                'Доля артиста': float(row['_artist_share']),
                'Общий доход трека': float(row['_track_licensee'])
            })
        
        pd.DataFrame(track_details).to_excel(
            writer, sheet_name='Детализация по трекам', index=False)
    
    output.seek(0)
    return output.getvalue()

@app.delete("/clear-all-splits/")
async def clear_all_splits(db: Session = Depends(get_db)):
    """Полная очистка всех сплитов"""
    try:
        db.query(SplitDB).delete()
        db.commit()
        return {"message": "Все сплиты успешно удалены", "status": "success"}
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка при очистке сплитов: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))
    
@app.get("/get-all-splits/")
async def get_all_splits(db: Session = Depends(get_db)):
    """Получение всех текущих сплитов"""
    try:
        splits = db.query(SplitDB).all()
        return {
            "splits": [
                {
                    "track_title": split.track_title,
                    "artist": split.artist_nickname,
                    "percentage": float(split.split_percentage)
                } 
                for split in splits
            ]
        }
    except Exception as e:
        logger.error(f"Ошибка получения сплитов: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.post("/add-split/")
async def create_track_splits(
    data: TrackSplitsRequest, 
    db: Session = Depends(get_db)
):
    """Сохранение распределения прав для трека"""
    try:
        # Проверка суммы процентов
        total = sum(Decimal(str(s.split_percentage)) for s in data.splits)
        if total > Decimal('100'):
            raise HTTPException(400, "Сумма процентов превышает 100%")
        
        # Удаляем старые сплиты
        db.query(SplitDB).filter(
            SplitDB.track_title == data.track_title
        ).delete()
        
        # Сохраняем новые сплиты
        for split in data.splits:
            artist = db.query(ArtistDB).filter(
                ArtistDB.nickname == split.nickname
            ).first()
            
            if not artist:
                artist = ArtistDB(nickname=split.nickname)
                db.add(artist)
                db.commit()
                db.refresh(artist)
            
            db.add(SplitDB(
                track_title=data.track_title,
                artist_nickname=split.nickname,
                split_percentage=Decimal(str(split.percentage))
            ))
        
        db.commit()
        return {
            "message": f"Сплиты для трека '{data.track_title}' сохранены",
            "applied_splits": {s.nickname: float(s.percentage) for s in data.splits}
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка сохранения сплитов: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))
    
def apply_splits_to_profits(row: pd.Series, track_splits: dict) -> pd.Series:
    """Применяет сплиты к строке данных"""
    try:
        track = str(row['_normalized_track_name']).strip().lower()
        artists = [str(a).strip().lower() for a in row.get('_parsed_performers', [])]
        
        if not track or not artists:
            return row
        
        if track not in track_splits:
            return row
        
        # Применяем все сплиты для трека
        for artist, percent in track_splits[track].items():
            artist_normalized = str(artist).strip().lower()
            if artist_normalized in artists:
                # Преобразуем процент и применяем
                percent_decimal = Decimal(str(percent)) / Decimal(100)
                row['royalty'] = Decimal(str(row['royalty'])) * percent_decimal
                logger.info(f"Applied split: {artist} {percent}% to {track}")
                
        return row
    except Exception as e:
        logger.error(f"Split application error: {e}")
        return row

def apply_splits(df: pd.DataFrame, db: Session) -> pd.DataFrame:
    """Добавляем информацию о сплитах в DataFrame"""
    df['_normalized_track_name'] = df.apply(get_track_name, axis=1)
    
    splits = db.query(SplitDB).all()
    track_splits = defaultdict(dict)
    for split in splits:
        track_splits[split.track_title][split.artist_nickname] = convert_to_decimal(split.split_percentage)
    
    for idx, row in df.iterrows():
        track = row['_normalized_track_name']
        if track in track_splits:
            for artist, percent in track_splits[track].items():
                df.at[idx, f'split_{artist}'] = percent
    
    return df

def add_profit_columns(df: pd.DataFrame, profit_data: dict) -> pd.DataFrame:
    """Добавляем колонки с финансовыми данными в DataFrame"""
    # Добавляем данные по трекам
    for track, values in profit_data['tracks'].items():
        mask = df['_normalized_track_name'] == track
        df.loc[mask, '_track_licensor'] = values['licensor']
        df.loc[mask, '_track_licensee'] = values['licensee']
    
    # Добавляем данные по артистам
    for artist, values in profit_data['artists'].items():
        def get_artist_share(row):
            if isinstance(row['_parsed_performers'], list) and artist in row['_parsed_performers']:
                track = row['_normalized_track_name']
                if track in profit_data['tracks']:
                    total = profit_data['tracks'][track]['licensee']
                    if total > 0:
                        return float(values['licensee'] / total * row['rights_holder_total'])
            return 0.0
        
        df[f'_share_{artist}'] = df.apply(get_artist_share, axis=1)
    
    return df

def calculate_artist_share(row: pd.Series, artist: str, profit_data: dict) -> Decimal:
    """Вычисляем долю конкретного артиста в конкретном треке"""
    if artist not in row['_parsed_performers']:
        return Decimal('0')
    
    track = row['_normalized_track_name']
    if track not in profit_data['tracks']:
        return Decimal('0')
    
    total_profit = profit_data['tracks'][track]['licensee']
    if total_profit == 0:
        return Decimal('0')
    
    artist_profit = profit_data['artists'][artist]['licensee']
    return (artist_profit / total_profit * row['rights_holder_total']).quantize(Decimal('0.01'))

def prepare_financial_summary(profit_data: dict) -> pd.DataFrame:
    """Подготавливаем сводную финансовую информацию"""
    # Основная сводка
    summary_data = {
        'Тип дохода': ['Прибыль платформы', 'Прибыль артистов', 'Общий доход'],
        'Сумма': [
            float(profit_data['totals']['licensor']),
            float(profit_data['totals']['licensee']),
            float(profit_data['totals']['licensor'] + profit_data['totals']['licensee'])
        ]
    }
    
    # Создаем отдельный DataFrame для основной сводки
    summary_df = pd.DataFrame(summary_data)
    
    # Данные по артистам (отдельный DataFrame)
    artists_data = []
    for artist, values in profit_data['artists'].items():
        artists_data.append({
            'Артист': artist,
            'Доля от платформы': float(values['licensor']),
            'Доля от прав': float(values['licensee']),
            'Общий доход': float(values['licensor'] + values['licensee'])
        })
    
    artists_df = pd.DataFrame(artists_data)
    
    # Возвращаем оба DataFrame (можно выбрать один или изменить логику вывода)
    return tuple (summary_df, artists_df)  # Или можно вернуть tuple (summary_df, artists_df)

@app.post("/test-split-application")
async def test_split_application(
    track_title: str = "Венера",
    artist: str = "Алиш",
    db: Session = Depends(get_db)
):
    try:
        # 1. Получаем сплит из БД
        split = db.query(SplitDB).filter(
            func.lower(SplitDB.track_title) == track_title.lower(),
            func.lower(SplitDB.artist_nickname) == artist.lower()
        ).first()
        
        if not split:
            return {"error": "Сплит не найден"}, 404
        
        # 2. Создаем тестовые данные
        test_data = {
            '_normalized_track_name': track_title,
            '_parsed_performers': [artist],
            'royalty': Decimal('10.00')
        }
        test_row = pd.Series(test_data)
        
        # 3. Формируем структуру сплитов
        track_splits = {
            track_title.lower(): {
                artist.lower(): split.split_percentage
            }
        }
        
        # 4. Применяем сплиты
        processed_row = apply_splits_to_profits(test_row.copy(), track_splits)
        
        return {
            "original_royalty": float(test_row['royalty']),
            "processed_royalty": float(processed_row['royalty']),
            "split_percentage": float(split.split_percentage),
            "expected": float(Decimal('10.00') * (split.split_percentage / Decimal(100))),
            "debug": {
                "track": track_title,
                "artist": artist,
                "normalization": {
                    "track": track_title.lower(),
                    "artist": artist.lower()
                }
            }
        }
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        return {"error": str(e)}, 500
    
@app.post("/test-db-write")
async def test_db_write(db: Session = Depends(get_db)):
    try:
        test = ArtistDB(nickname="test_artist")
        db.add(test)
        db.commit()  # Явное сохранение
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=str(e))
    
@app.get("/debug-check")
async def debug_check(db: Session = Depends(get_db)):
    try:
        # Проверка подключения
        db.execute(text("SELECT 1"))
        
        # Попытка создания тестовой записи
        test_artist = ArtistDB(nickname="debug_artist")
        db.add(test_artist)
        db.commit()
        
        # Проверка что запись существует
        exists = db.query(ArtistDB).filter_by(nickname="debug_artist").first()
        return {
            "connection": "ok",
            "record_added": bool(exists),
            "record_id": exists.id if exists else None
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/db-status")
async def db_status():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "details": str(e)}

@app.post("/generate-reports/")
async def generate_reports(
    reports: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        # 1. Загрузка и обработка данных
        processed_dfs = []
        for report in reports:
            df = await process_report(report, db)
            df['_normalized_track_name'] = df.apply(get_track_name, axis=1)
            df = df[df['_normalized_track_name'] != "Unknown Track"]
            processed_dfs.append(df)
        
        combined_df = pd.concat(processed_dfs, ignore_index=True).fillna(Decimal('0'))
        
        # 2. Единый расчет всех прибылей
        profit_data = calculate_profits(combined_df, db)
        
        # 3. Добавляем финансовые данные в основной DataFrame
        combined_df = add_profit_columns(combined_df, profit_data)
        
        # 4. Подготовка данных для артистов
        artist_reports = {}
        for artist, profits in profit_data['artists'].items():
            mask = combined_df['_parsed_performers'].apply(
                lambda x: artist in x if isinstance(x, list) else False
            )
            artist_data = combined_df[mask].copy()
            
            # Добавляем расчет доли артиста
            artist_data['_artist_share'] = artist_data.apply(
                lambda row: calculate_artist_share(row, artist, profit_data),
                axis=1
            )
            
            artist_reports[artist] = generate_artist_report(artist_data, artist, profits)
        
        # 5. Создание архива
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                # Отчеты артистов
                for artist, report in artist_reports.items():
                    clean_name = re.sub(r'[^\w]', '_', artist)
                    zipf.writestr(f"artists/{clean_name}.xlsx", report)
                
                # Общий отчет
                full_report = io.BytesIO()
                with pd.ExcelWriter(full_report, engine='xlsxwriter') as writer:
                    # Сводка по трекам
                    combined_df.to_excel(writer, sheet_name='Треки', index=False)
                    
                    # Финансовая сводка
                    summary_df = pd.DataFrame({
                        'Тип дохода': ['Прибыль платформы', 'Прибыль артистов', 'Общий доход'],
                        'Сумма': [
                            float(profit_data['totals']['licensor']),
                            float(profit_data['totals']['licensee']),
                            float(profit_data['totals']['licensor'] + profit_data['totals']['licensee'])
                        ]
                    })
                    summary_df.to_excel(writer, sheet_name='Финансовая сводка', index=False)
                    
                    # Детализация по артистам
                    artists_data = []
                    for artist, values in profit_data['artists'].items():
                        artists_data.append({
                            'Артист': artist,
                            'Доля от платформы': float(values['licensor']),
                            'Доля от прав': float(values['licensee']),
                            'Общий доход': float(values['licensor'] + values['licensee'])
                        })
                    
                    pd.DataFrame(artists_data).to_excel(
                        writer, sheet_name='Детализация по артистам', index=False)
                
                zipf.writestr("00_FULL_REPORT.xlsx", full_report.getvalue())

        background_tasks.add_task(os.unlink, tmp_zip.name)
        return FileResponse(tmp_zip.name, filename="reports.zip")

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    try:
        Base.metadata.create_all(bind=engine)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Failed to start: {str(e)}")
        raise