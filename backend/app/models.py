
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Numeric
from sqlalchemy.orm import relationship
from backend.app.database import get_db, Base, engine
from datetime import datetime
import numbers


    #releases = relationship("ReleaseDB", back_populates="artist")
    #splits = relationship("SplitDB", back_populates="artist")
    #final_reports = relationship("FinalReportDB", back_populates="artist")
    #individual_reports = relationship("IndividualReportDB", back_populates="artist")
    #combined_reports_artist1 = relationship("CombinedReportDB", foreign_keys="[CombinedReportDB.artist1_id]", back_populates="artist1")
    #combined_reports_artist2 = relationship("CombinedReportDB", foreign_keys="[CombinedReportDB.artist2_id]", back_populates="artist2")

# class ReleaseDB(Base):
#     __tablename__ = "releases"

#     id = Column(Integer, primary_key=True, index=True)
#     artist_id = Column(Integer, ForeignKey("artists.id"))
#     release_name = Column(String)
#     rights_share = Column(Float)
#     release_date = Column(Date)
#     isrc = Column(String, nullable=True)
#     upc = Column(String, nullable=True)

#     # Relationships
#     artist = relationship("ArtistDB", back_populates="releases")
#     splits = relationship("SplitDB", back_populates="release")
#     reports = relationship("AggregatorReportDB", back_populates="release")
#     individual_reports = relationship("IndividualReportDB", back_populates="release")
#     combined_reports = relationship("CombinedReportDB", back_populates="release")

class ArtistDB(Base):
    __tablename__ = "artists"
    id = Column(Integer, primary_key=True, index=True)
    nickname = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Только актуальные отношения
    splits = relationship("SplitDB", back_populates="artist")

class SplitDB(Base):
    __tablename__ = "splits"
    id = Column(Integer, primary_key=True, index=True)
    track_title = Column(String, index=True)
    artist_nickname = Column(String, ForeignKey("artists.nickname"))
    split_percentage = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    artist = relationship("ArtistDB", back_populates="splits")

    
    #release = relationship("ReleaseDB", back_populates="splits")

# class IndividualReportDB(Base):
#     __tablename__ = "individual_reports"
    
#     id = Column(Integer, primary_key=True, index=True)
#     artist_id = Column(Integer, ForeignKey("artists.id"))
#     release_id = Column(Integer, ForeignKey("releases.id"))
#     split_percentage = Column(Float)
#     created_at = Column(DateTime, default=datetime.utcnow)
    
#     artist = relationship("ArtistDB", back_populates="individual_reports")
#     release = relationship("ReleaseDB", back_populates="individual_reports")

# class CombinedReportDB(Base):
#     __tablename__ = "combined_reports"

#     id = Column(Integer, primary_key=True, index=True)
#     artist1_id = Column(Integer, ForeignKey("artists.id"))
#     artist2_id = Column(Integer, ForeignKey("artists.id"))
#     release_id = Column(Integer, ForeignKey("releases.id"))
#     split_percentage_artist1 = Column(Float)
#     split_percentage_artist2 = Column(Float)
#     created_at = Column(DateTime, default=datetime.utcnow)

#     # Relationships
#     artist1 = relationship("ArtistDB", foreign_keys=[artist1_id], back_populates="combined_reports_artist1")
#     artist2 = relationship("ArtistDB", foreign_keys=[artist2_id], back_populates="combined_reports_artist2")
#     release = relationship("ReleaseDB", back_populates="combined_reports")

# # Остальные существующие модели остаются без изменений
# class AggregatorReportDB(Base):
#     __tablename__ = "aggregator_reports"

#     id = Column(Integer, primary_key=True, index=True)
#     aggregator_name = Column(String)
#     release_id = Column(Integer, ForeignKey("releases.id"))
#     report_data = Column(JSON)
#     upload_date = Column(DateTime, default=datetime.utcnow)
    
#     release = relationship("ReleaseDB", back_populates="reports")

# class FinalReportDB(Base):
#     __tablename__ = "final_reports"

#     id = Column(Integer, primary_key=True, index=True)
#     artist_id = Column(Integer, ForeignKey("artists.id"))
#     report_period = Column(String)
#     total_revenue = Column(Float)
#     total_royalty = Column(Float)
#     artist_share = Column(Float)
#     artist_revenue = Column(Float)

#     artist = relationship("ArtistDB", back_populates="final_reports")