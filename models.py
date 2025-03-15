from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class ArtistDB(Base):
    __tablename__ = "artists"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)  # ФИО артиста
    contact_info = Column(String, nullable=True)  # Контактная информация

    releases = relationship("ReleaseDB", back_populates="artist")

class ReleaseDB(Base):
    __tablename__ = "releases"

    id = Column(Integer, primary_key=True, index=True)
    artist_id = Column(Integer, ForeignKey("artists.id"))  # Связь с артистом
    release_name = Column(String)  # Название релиза
    rights_share = Column(Float)  # Доля прав артиста
    release_date = Column(Date)  # Дата релиза
    isrc = Column(String, nullable=True)  # ISRC
    upc = Column(String, nullable=True)  # UPC

    artist = relationship("ArtistDB", back_populates="releases")

class QuarterlyReportDB(Base):
    __tablename__ = "quarterly_reports"

    id = Column(Integer, primary_key=True, index=True)
    release_id = Column(Integer, ForeignKey("releases.id"))  # Связь с релизом
    total_revenue = Column(Float)  # Общая выручка
    sales_quantity = Column(Integer)  # Количество продаж
    royalty_amount = Column(Float)  # Сумма роялти
    report_period = Column(String)  # Период отчета