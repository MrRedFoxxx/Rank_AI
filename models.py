from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# Таблица для избранных статей
favorites = Table(
    'favorites',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('article_id', Integer, ForeignKey('articles.id'), primary_key=True)
)

# Таблица для подписчиков
followers = Table(
    'followers',
    Base.metadata,
    Column('follower_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('followed_id', Integer, ForeignKey('users.id'), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    surname = Column(String)
    login = Column(String, unique=True)
    password = Column(String)  # Will store hashed password
    rating = Column(Float, default=0.0)
    type = Column(String, default='user')  # 'user' or 'admin'
    about = Column(String, nullable=True)
    like = Column(String, default='[]')  # JSON string for favorite articles

    # Отношения
    articles = relationship("Article", back_populates="user")
    favorite_articles = relationship("Article", secondary=favorites, back_populates="favorited_by")
    
    # Подписчики
    followers = relationship(
        'User', secondary=followers,
        primaryjoin=(followers.c.followed_id == id),
        secondaryjoin=(followers.c.follower_id == id),
        backref='following'
    )

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    preview_image = Column(String)
    authors = Column(String)
    abstract = Column(Text)
    keywords = Column(String)
    rating = Column(Float)
    file_path = Column(String)
    category = Column(Text)
    date = Column(Text)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    # Отношения
    user = relationship("User", back_populates="articles")
    favorited_by = relationship("User", secondary=favorites, back_populates="favorite_articles") 