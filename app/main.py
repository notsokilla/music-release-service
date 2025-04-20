from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import io
import re
import tempfile
import zipfile
import os
import logging
from pydantic import BaseModel
from decimal import Decimal, getcontext
from collections import defaultdict
from app.database import get_db, Base, engine
from app.models import ArtistDB, SplitDB
from fastapi.background import BackgroundTasks
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(docs_url="/docs", redoc_url="/redoc")
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://music-release-service-front.onrender.com/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORT_COLUMNS = [
    '_normalized_track_name',
    'gross',
    'net',
    'license_fee',
    'rights_holder_total',
    'комиссия_лицензиата',
    'правообладатель_вознаграждение_итого',
    '_parsed_performers'
]

FINANCIAL_COLUMNS = [
    'gross', 
    'net', 
    'license_fee',
    'rights_holder_total', 
    'комиссия_лицензиата',
    'правообладатель_вознаграждение_итого'
]

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

# Вспомогательные функции
def get_track_name(row: pd.Series) -> str:
    """Возвращает название трека из различных колонок"""
    for col in ['track_title', 'наименование', 'title', 'song_name', 'Title']:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return "Unknown Track"

def convert_to_decimal(value):
    """Безопасное преобразование в float с обработкой списков"""
    if isinstance(value, (list, np.ndarray)):
        value = value[0] if len(value) > 0 else 0
    try:
        return float(Decimal(str(float(value))))
    except (TypeError, ValueError):
        return 0.0

def prepare_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Приведение финансовых колонок к float"""
    money_cols = [
        'gross', 'net', 'license_fee',
        'rights_holder_total', 'комиссия_лицензиата',
        'правообладатель_вознаграждение_итого'
    ]
    
    for col in money_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].apply(
                lambda x: convert_to_decimal(x[0]) 
                if isinstance(x, (list, np.ndarray)) 
                else convert_to_decimal(x)
            )
        else:
            df.loc[:, col] = 0.0
    return df

def parse_performers(source: str, raw_artists: str) -> List[str]:
    """Парсинг исполнителей с сохранением всех спецсимволов"""
    if pd.isna(raw_artists):
        return []
    
    raw_artists = str(raw_artists).strip()
    if not raw_artists:
        return []
    
    if 'rpm' in source.lower():
        # Для RPM формата: сохраняем оригинальные имена с ролью performer
        performers = []
        current = []
        paren_level = 0
        
        for char in raw_artists:
            if char == '(':
                paren_level += 1
                current.append(char)
            elif char == ')':
                paren_level -= 1
                current.append(char)
            elif char == ',' and paren_level == 0:
                part = ''.join(current).strip()
                if '(performer)' in part.lower():
                    name = part.split('(')[0].strip()
                    performers.append(name)
                current = []
            else:
                current.append(char)
        
        # Добавляем последнюю часть
        if current:
            part = ''.join(current).strip()
            if '(performer)' in part.lower():
                name = part.split('(')[0].strip()
                performers.append(name)
        
        return performers
    else:
        # Для обычного формата: разделяем по запятым, сохраняя все символы
        performers = []
        current = []
        in_slash = False
        
        for char in raw_artists:
            if char == '/':
                in_slash = not in_slash
                current.append(char)
            elif char == ',' and not in_slash:
                performer = ''.join(current).strip()
                if performer:
                    performers.append(performer)
                current = []
            else:
                current.append(char)
        
        # Добавляем последнего исполнителя
        if current:
            performer = ''.join(current).strip()
            if performer:
                performers.append(performer)
        
        return performers

async def process_report(file: UploadFile, db: Session) -> pd.DataFrame:
    """Обработка загруженного отчета с сохранением спецсимволов"""
    contents = await file.read()
    
    try:
        # Чтение файла
        if file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {str(e)}")
        raise HTTPException(400, "Некорректный формат файла")

    # Нормализация названий колонок (только для системных целей)
    df.columns = [re.sub(r'[^\w]', '_', col.strip().lower()) for col in df.columns]
    
    # Создание недостающих финансовых колонок
    for col in FINANCIAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    # Нормализация названий треков
    df['_normalized_track_name'] = df.apply(get_track_name, axis=1)
    
    # Парсинг исполнителей с сохранением оригинальных имен
    source_type = file.filename.lower()
    artist_column = next((col for col in df.columns if col in ['artists', 'исполнитель']), None)
    
    if artist_column:
        df['_parsed_performers'] = df[artist_column].apply(
            lambda x: parse_performers(source_type, x)
        )
    else:
        df['_parsed_performers'] = [[] for _ in range(len(df))]
    
    return df

def calculate_artist_profit(row: dict, artist: str, artist_names: List[str]) -> Dict[str, Decimal]:
    """Оригинальная формула расчета с Decimal с обработкой ошибок"""
    try:
        # Получаем долю артиста, преобразуем в Decimal
        split_value = str(row.get(f'split_{artist}', '0')).strip()
        artist_share = Decimal(split_value if split_value.replace('.', '', 1).isdigit() else '0') / Decimal('100')
        
        # Получаем финансовые показатели
        gross = Decimal(str(row.get('gross', 0)))
        net = Decimal(str(row.get('net', 0)))
        license_fee = Decimal(str(row.get('license_fee', 0)))
        licensee_commission = Decimal(str(row.get('комиссия_лицензиата', 0)))
        rights_holder_total = Decimal(str(row.get('rights_holder_total', 0)))
        licensor_total = Decimal(str(row.get('правообладатель_вознаграждение_итого', 0)))

        license_value = gross - net + license_fee + licensee_commission
        base_licensor = net + rights_holder_total + licensor_total
        
        num_artists = Decimal(str(len(artist_names))) if artist_names else Decimal('1')
        
        adjustment = adjustment2 = Decimal('0')
        if artist_share > Decimal('0') and num_artists > Decimal('0'):
            adjustment = (license_value - num_artists * license_value * artist_share) / num_artists
            adjustment2 = (base_licensor - num_artists * base_licensor * artist_share) / num_artists
        
        return {
            'licensee': license_value * artist_share + adjustment + adjustment2,
            'licensor': base_licensor * artist_share
        }
    except Exception as e:
        logger.error(f"Ошибка расчета прибыли для артиста {artist}: {str(e)}")
        return {
            'licensee': Decimal('0'),
            'licensor': Decimal('0')
        }

def generate_artist_report(df: pd.DataFrame, artist: str) -> bytes:
    """Генерация отчета с обработкой ошибок"""
    try:
        split_col = f'split_{artist}'
        
        # Инициализируем колонку split, если её нет
        if split_col not in df.columns:
            df[split_col] = '0'
        
        # Конвертируем в строку и очищаем значения
        df[split_col] = df[split_col].astype(str).str.strip()
        # Заменяем некорректные значения на '0'
        df[split_col] = df[split_col].apply(lambda x: x if x.replace('.', '', 1).isdigit() else '0')
        
        mask = df['_parsed_performers'].apply(
            lambda x: artist in x if isinstance(x, list) else False
        )
        artist_df = df[mask].copy()

        total_licensor = Decimal('0')
        total_licensee = Decimal('0')
        
        for _, row in artist_df.iterrows():
            row_dict = row.to_dict()
            artists = row.get('_parsed_performers', [])
            profit = calculate_artist_profit(row_dict, artist, artists)
            
            total_licensor += Decimal(str(profit['licensor']))
            total_licensee += Decimal(str(profit['licensee']))

        # Создаем отчет
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Лист с данными
            report_columns = [
                '_normalized_track_name',
                'gross', 'net', 'license_fee',
                'rights_holder_total',
                'комиссия_лицензиата',
                'правообладатель_вознаграждение_итого',
                '_parsed_performers',
                split_col
            ]
            
            available_cols = [col for col in report_columns if col in artist_df.columns]
            artist_df[available_cols].to_excel(
                writer,
                sheet_name='Данные',
                index=False,
                float_format="%.2f"
            )
            
            # Лист с финансами
            workbook = writer.book
            worksheet = workbook.add_worksheet('Финансы')
            bold = workbook.add_format({'bold': True})
            money_format = workbook.add_format({'num_format': '#,##0.00'})
            
            worksheet.write(0, 0, "Прибыль лицензиата:", bold)
            worksheet.write(0, 1, float(total_licensee), money_format)
            worksheet.write(1, 0, "Прибыль лицензиара:", bold)
            worksheet.write(1, 1, float(total_licensor), money_format)
            
            worksheet.set_column('A:A', 25)
            worksheet.set_column('B:B', 15, money_format)

        output.seek(0)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Ошибка генерации отчета для {artist}: {str(e)}")
        # Возвращаем пустой отчет в случае ошибки
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame().to_excel(writer, index=False)
        output.seek(0)
        return output.getvalue()

@app.post("/add-split/")
async def create_track_splits(
    data: TrackSplitsRequest, 
    db: Session = Depends(get_db)
):
    """Сохранение распределения прав для трека"""
    logger.info(f"Начало обработки сплитов для трека: {data.track_title}")
    
    try:
        # Логирование входящих данных
        logger.debug(f"Получены сплиты: {[(s.nickname, s.percentage) for s in data.splits]}")

        if not data.splits:
            logger.warning("Сплиты не предоставлены, пытаюсь определить автоматически")
            track_artists = db.query(SplitDB.artist_nickname).filter(
                SplitDB.track_title == data.track_title
            ).all()
            
            artist_names = [a[0] for a in track_artists] if track_artists else []
            logger.debug(f"Найдены существующие артисты для трека: {artist_names}")
            
            if not artist_names:
                logger.error("Не удалось определить артистов для автоматического распределения")
                return {"message": "Нет артистов для автоматического распределения"}
            
            default_percent = 100.0 / len(artist_names)
            data.splits = [
                SplitRequest(nickname=name, percentage=float(default_percent))
                for name in artist_names
            ]
            logger.info(f"Автоматически распределены доли: {[(s.nickname, s.percentage) for s in data.splits]}")

        total_percent = sum(s.percentage for s in data.splits)
        logger.debug(f"Общий процент сплитов: {total_percent}%")
        
        if total_percent > 100:
            logger.error(f"Сумма процентов превышает 100%: {total_percent}%")
            raise HTTPException(400, "Сумма процентов превышает 100%")

        # Удаление старых сплитов
        deleted = db.query(SplitDB).filter(SplitDB.track_title == data.track_title).delete()
        logger.info(f"Удалено старых сплитов: {deleted}")

        # Сохранение новых сплитов
        for split in data.splits:
            logger.debug(f"Обработка артиста: {split.nickname} ({split.percentage}%)")
            
            artist = db.query(ArtistDB).filter(ArtistDB.nickname == split.nickname).first()
            if not artist:
                logger.info(f"Артист {split.nickname} не найден, создаю нового")
                artist = ArtistDB(nickname=split.nickname)
                db.add(artist)
                db.commit()
                db.refresh(artist)
                logger.debug(f"Создан новый артист: {artist.id} {artist.nickname}")
            
            new_split = SplitDB(
                track_title=data.track_title,
                artist_nickname=split.nickname,
                split_percentage=Decimal(str(split.percentage)))
            
            db.add(new_split)
            logger.debug(f"Добавлен сплит: {new_split}")
        
        db.commit()
        logger.info(f"Сплиты для трека '{data.track_title}' успешно сохранены")
        
        return {
            "message": f"Сплиты для трека '{data.track_title}' сохранены",
            "applied_splits": {s.nickname: s.percentage for s in data.splits}
        }
    
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка сохранения сплитов: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))
    

@app.get("/get-splits/", response_model=List[Dict])
async def get_all_splits(db: Session = Depends(get_db)):
    """Получить все распределения прав (новый эндпоинт)"""
    splits = db.query(SplitDB).all()
    return [
        {
            "track_title": split.track_title,
            "artist": split.artist_nickname,
            "percentage": float(split.split_percentage)
        }
        for split in splits
    ]

@app.delete("/delete-splits/")
async def delete_all_splits(db: Session = Depends(get_db)):
    """Удалить ВСЕ распределения прав (новый эндпоинт)"""
    try:
        db.query(SplitDB).delete()
        db.commit()
        return {"message": "Все сплиты успешно удалены"}
    except Exception as e:
        db.rollback()
        logger.error(f"Ошибка удаления: {str(e)}")
        raise HTTPException(500, detail="Ошибка при удалении сплитов")

def apply_splits(df: pd.DataFrame, db: Session) -> pd.DataFrame:
    df = df.copy()
    
    # Получаем все сплиты из БД (сохраняем оригинальные имена)
    db_splits = db.query(SplitDB).all()
    
    # Создаем словарь: {track_title: {original_artist_name: percentage}}
    track_splits = defaultdict(dict)
    for split in db_splits:
        track_splits[split.track_title][split.artist_nickname] = float(split.split_percentage)
    
    # Обрабатываем каждый трек
    for _, row in df.iterrows():
        track_title = row['_normalized_track_name']
        performers = row.get('_parsed_performers', [])
        
        if track_title in track_splits:
            total_percent = 0.0
            for artist, percent in track_splits[track_title].items():
                col_name = f'split_{artist}'  # Сохраняем оригинальное имя
                df.at[_, col_name] = percent
                total_percent += percent
            
            df.at[_, 'label_share'] = max(0.0, 100.0 - total_percent)
        elif performers:
            share = 100.0 / len(performers)
            for artist in performers:
                col_name = f'split_{artist}'  # Сохраняем оригинальное имя
                df.at[_, col_name] = share
            df.at[_, 'label_share'] = 0.0
    
    return df

@app.post("/generate-reports/")
async def generate_reports(
    reports: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        processed_dfs = []
        for report in reports:
            df = await process_report(report, db)
            processed_dfs.append(df)
        
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        combined_df = apply_splits(combined_df, db)
        
        # Генерация ZIP
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            zip_filename = tmp_zip.name
        
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            # Персональные отчеты
            all_artists = {a for performers in combined_df['_parsed_performers'] for a in performers}
            for artist in all_artists:
                report_content = generate_artist_report(combined_df, artist)
                zipf.writestr(f"{artist}_report.xlsx", report_content)
            
            # Сводный отчет
            summary_output = io.BytesIO()
            with pd.ExcelWriter(summary_output, engine='xlsxwriter') as writer:
                combined_df.to_excel(writer, sheet_name="Все данные", index=False)
                
                workbook = writer.book
                worksheet = workbook.add_worksheet("Финансы")
                bold = workbook.add_format({'bold': True})
                money_format = workbook.add_format({'num_format': '#,##0.00'})
                
                # Расчет общей прибыли
                total_licensee = Decimal('0')
                total_licensor = Decimal('0')
                
                # Группируем по трекам для корректного расчета
                for track_name, group in combined_df.groupby('_normalized_track_name'):
                    # Получаем список исполнителей для трека
                    performers = []
                    for plist in group['_parsed_performers']:
                        performers.extend(plist)
                    performers = list(set(performers))  # Уникальные исполнители
                    
                    # Суммируем финансовые показатели по треку
                    track_gross = group['gross'].sum()
                    track_net = group['net'].sum()
                    track_license_fee = group['license_fee'].sum()
                    track_licensee_commission = group['комиссия_лицензиата'].sum()
                    track_rights_holder_total = group['rights_holder_total'].sum()
                    track_licensor_total = group['правообладатель_вознаграждение_итого'].sum()
                    
                    # Расчет по оригинальной формуле для трека
                    license_value = Decimal(str(track_gross)) - Decimal(str(track_net)) + \
                                  Decimal(str(track_license_fee)) + Decimal(str(track_licensee_commission))
                    base_licensor = Decimal(str(track_net)) + \
                                   Decimal(str(track_rights_holder_total)) + \
                                   Decimal(str(track_licensor_total))
                    
                    num_artists = Decimal(str(len(performers))) if performers else Decimal('1')
                    
                    # Распределяем по исполнителям
                    for artist in performers:
                        artist_col = f'split_{artist}'
                        if artist_col in group.columns:
                            artist_share = Decimal(str(group[artist_col].iloc[0])) / Decimal('100')
                        else:
                            # Если нет сплита, распределяем поровну
                            artist_share = Decimal('1') / num_artists
                        
                        # Расчет по оригинальной формуле
                        adjustment = adjustment2 = Decimal('0')
                        if artist_share > Decimal('0') and num_artists > Decimal('0'):
                            adjustment = (license_value - num_artists * license_value * artist_share) / num_artists
                            adjustment2 = (base_licensor - num_artists * base_licensor * artist_share) / num_artists
                        
                        total_licensee += license_value * artist_share + adjustment + adjustment2
                        total_licensor += base_licensor * artist_share
                
                worksheet.write(0, 0, "Общая прибыль лицензиата (включая долю лейбла):", bold)
                worksheet.write(0, 1, float(total_licensee), money_format)
                worksheet.write(1, 0, "Общая прибыль лицензиара:", bold)
                worksheet.write(1, 1, float(total_licensor), money_format)
                
                worksheet.set_column('A:A', 35)  # Увеличиваем ширину для длинного заголовка
                worksheet.set_column('B:B', 15, money_format)
            
            summary_output.seek(0)
            zipf.writestr("00_FULL_REPORT.xlsx", summary_output.getvalue())
        
        background_tasks.add_task(os.unlink, zip_filename)
        return FileResponse(zip_filename, media_type='application/zip', filename="reports.zip")

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))
    
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)