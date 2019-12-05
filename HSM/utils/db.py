from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, JSON, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, sessionmaker
from utils.config import SQLALCHEMY_URI

Base = declarative_base()


class DataAccessLayer:

    def __init__(self):
        self.engine = None
        self.conn_string = SQLALCHEMY_URI

    def connect(self):
        self.engine = create_engine(self.conn_string, echo=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)


dal = DataAccessLayer()


class Data(Base):
    __tablename__ = 'data'
    id = Column(Integer, primary_key=True, index=True)
    filter_feature = Column(String(10000), nullable=True)
    validation = Column(Integer)

    support_data = relationship("SupportData", uselist=False, back_populates="data")


class SupportData(Base):
    __tablename__ = 'support_data'
    id = Column(Integer, primary_key=True, index=True)
    support_data = Column(JSON)

    data_id = Column(Integer, ForeignKey('data.id'), nullable=False)
    data = relationship("Data", back_populates="support_data")
