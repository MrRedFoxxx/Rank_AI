from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import models
from sqlalchemy import select, text
from models import Article
from typing import Optional
import json

# Create async engine
SQLALCHEMY_DATABASE_URL = "sqlite+aiosqlite:///./db.db"
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create async session factory
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Initialize database
async def init_db():
    async with engine.begin() as conn:
        # Только создаем таблицы, если они не существуют
        await conn.run_sync(models.Base.metadata.create_all)

# Dependency to get DB session
async def get_db():
    async with async_session() as session:
        yield session

# User operations
async def get_user_by_login(db: AsyncSession, login: str):
    result = await db.execute(
        text("SELECT * FROM users WHERE login = :login"),
        {"login": login}
    )
    row = result.first()
    if row is None:
        return None
    # Convert row to dictionary with correct column order
    return {
        "id": row[0],
        "name": row[1],
        "surname": row[2],
        "login": row[3],
        "password": row[4],
        "rating": row[5],
        "role": row[6],
        "about": row[7],
        "like": row[8]
    }

async def get_user_by_id(db: AsyncSession, user_id: int):
    result = await db.execute(
        text("SELECT * FROM users WHERE id = :id"),
        {"id": user_id}
    )
    row = result.first()
    if row is None:
        return None
    # Convert row to dictionary with correct column order
    return {
        "id": row[0],
        "name": row[1],
        "surname": row[2],
        "login": row[3],
        "password": row[4],
        "rating": row[5],
        "role": row[6],
        "about": row[7],
        "like": row[8]
    }

async def create_user(
    db: AsyncSession,
    login: str,
    password: str,
    name: str,
    surname: str,
    about: str = None,
    type: str = 'user'
):
    try:
        # Создаем пользователя
        query = text("""
        INSERT INTO users (login, password, name, surname, rating, type, about, like)
        VALUES (:login, :password, :name, :surname, 0.0, :type, :about, '[]')
        """)
        values = {
            "login": login,
            "password": password,
            "name": name,
            "surname": surname,
            "type": type,
            "about": about
        }
        
        result = await db.execute(query, values)
        await db.commit()
        
        # Получаем ID последней вставленной записи
        last_id = result.lastrowid
        
        # Получаем созданного пользователя
        result = await db.execute(
            text("SELECT * FROM users WHERE id = :id"),
            {"id": last_id}
        )
        row = result.first()
        
        if row is None:
            return None
            
        return {
            "id": last_id,
            "name": row[1],
            "surname": row[2],
            "login": row[3],
            "password": row[4],
            "rating": float(row[5]) if row[5] is not None else 0.0,
            "role": row[6],
            "about": row[7],
            "like": row[8]
        }
    except Exception as e:
        await db.rollback()
        raise e

async def update_user(db: AsyncSession, user_id: int, update_data: dict):
    # Создаем строку с обновлениями
    updates = ", ".join([f"{k} = :{k}" for k in update_data.keys()])
    query = text(f"UPDATE users SET {updates} WHERE id = :id RETURNING *")
    
    # Добавляем id в параметры
    params = dict(update_data)
    params["id"] = user_id
    
    result = await db.execute(query, params)
    await db.commit()
    return result.first()

# Article operations
async def get_user_articles(
    db: AsyncSession,
    user_id: int,
    limit: int = 10,
    offset: int = 0,
    sort_by: str = "date",
    sort_order: str = "desc",
    category: Optional[str] = None
):
    base_query = "SELECT * FROM articles WHERE user_id = :user_id"
    if category:
        base_query += " AND category = :category"
    
    query = text(f"{base_query} ORDER BY {sort_by} {sort_order} LIMIT :limit OFFSET :offset")
    
    params = {
        "user_id": user_id,
        "limit": limit,
        "offset": offset
    }
    if category:
        params["category"] = category
        
    result = await db.execute(query, params)
    return result.all()

async def create_article(
    db: AsyncSession,
    title: str,
    preview_image: Optional[str],
    authors: str,
    abstract: str,
    keywords: Optional[str],
    file_path: str,
    category: str,
    date: str,
    user_id: int
) -> Article:
    db_article = Article(
        title=title,
        preview_image=preview_image,
        authors=authors,
        abstract=abstract,
        keywords=keywords,
        rating=0.0,  # Initial rating
        file_path=file_path,
        category=category,
        date=date,
        user_id=user_id
    )
    db.add(db_article)
    await db.commit()
    await db.refresh(db_article)
    return db_article

async def update_article(
    db: AsyncSession,
    article_id: int,
    title: Optional[str] = None,
    preview_image: Optional[str] = None,
    abstract: Optional[str] = None,
    keywords: Optional[str] = None,
    category: Optional[str] = None
) -> Optional[Article]:
    result = await db.execute(
        select(Article).where(Article.id == article_id)
    )
    article = result.scalar_one_or_none()
    
    if article:
        if title:
            article.title = title
        if preview_image:
            article.preview_image = preview_image
        if abstract:
            article.abstract = abstract
        if keywords:
            article.keywords = keywords
        if category:
            article.category = category
            
        await db.commit()
        await db.refresh(article)
        
    return article

async def get_article_by_id(db: AsyncSession, article_id: int):
    result = await db.execute(
        text("SELECT * FROM articles WHERE id = :id"),
        {"id": article_id}
    )
    return result.first()

async def search_articles(db: AsyncSession, query: str):
    print(f"Database search query: {query}")  # Debug log
    try:
        sql_query = text("""
            SELECT * FROM articles 
            WHERE LOWER(title) LIKE LOWER(:query) 
            OR LOWER(abstract) LIKE LOWER(:query)
            OR LOWER(authors) LIKE LOWER(:query)
            OR LOWER(keywords) LIKE LOWER(:query)
            OR LOWER(category) LIKE LOWER(:query)
            ORDER BY rating DESC, date DESC
        """)
        print(f"SQL Query: {sql_query}")  # Debug log
        result = await db.execute(sql_query, {"query": f"%{query}%"})
        articles = result.all()
        print(f"Database results: {articles}")  # Debug log
        return articles
    except Exception as e:
        print(f"Database search error: {str(e)}")
        return []

# Операции с избранными статьями
async def add_to_favorites(db: AsyncSession, user_id: int, article_id: int):
    # Получаем текущий список избранных статей
    user = await get_user_by_id(db, user_id)
    if not user:
        raise Exception("User not found")
    
    # Парсим текущий список избранных
    favorites = json.loads(user["like"] or "[]")
    
    # Добавляем новую статью, если её ещё нет
    if article_id not in favorites:
        favorites.append(article_id)
        
        # Обновляем список избранных
        await update_user(db, user_id, {"like": json.dumps(favorites)})
        return True
    return False

async def remove_from_favorites(db: AsyncSession, user_id: int, article_id: int):
    # Получаем текущий список избранных статей
    user = await get_user_by_id(db, user_id)
    if not user:
        raise Exception("User not found")
    
    # Парсим текущий список избранных
    favorites = json.loads(user["like"] or "[]")
    
    # Удаляем статью из списка
    if article_id in favorites:
        favorites.remove(article_id)
        
        # Обновляем список избранных
        await update_user(db, user_id, {"like": json.dumps(favorites)})
        return True
    return False

async def get_user_favorites(db: AsyncSession, user_id: int):
    user = await get_user_by_id(db, user_id)
    if not user:
        return []
    
    # Парсим список избранных статей
    favorites = json.loads(user["like"] or "[]")
    
    # Получаем информацию о каждой статье
    articles = []
    for article_id in favorites:
        article = await get_article_by_id(db, article_id)
        if article:
            articles.append(article)
    
    return articles

async def get_user_stats(db: AsyncSession, user_id: int) -> dict:
    """Получение статистики пользователя"""
    try:
        # Получаем количество статей пользователя
        articles_count_query = text("""
            SELECT COUNT(*) FROM articles WHERE user_id = :user_id
        """)
        articles_count = await db.execute(articles_count_query, {"user_id": user_id})
        articles_count = articles_count.scalar() or 0

        # Получаем количество избранных статей из поля like
        favorites_count_query = text("""
            SELECT json_array_length(like) FROM users WHERE id = :user_id
        """)
        favorites_count = await db.execute(favorites_count_query, {"user_id": user_id})
        favorites_count = favorites_count.scalar() or 0

        # Получаем средний рейтинг статей пользователя
        avg_rating_query = text("""
            SELECT AVG(rating) FROM articles WHERE user_id = :user_id
        """)
        avg_rating = await db.execute(avg_rating_query, {"user_id": user_id})
        avg_rating = avg_rating.scalar() or 0.0

        return {
            "articles_count": articles_count,
            "favorites_count": favorites_count,
            "avg_rating": round(float(avg_rating), 2)
        }
    except Exception as e:
        print(f"Error getting user stats: {str(e)}")
        return {
            "articles_count": 0,
            "favorites_count": 0,
            "avg_rating": 0.0
        }

async def get_recommended_articles(db: AsyncSession, user_id: int, limit: int = 10):
    # Получаем рекомендованные статьи на основе категорий статей пользователя
    query = text("""
    WITH user_categories AS (
        SELECT DISTINCT category 
        FROM articles 
        WHERE user_id = :user_id
        UNION
        SELECT DISTINCT a.category
        FROM articles a
        JOIN favorites f ON f.article_id = a.id
        WHERE f.user_id = :user_id
    )
    SELECT DISTINCT a.* 
    FROM articles a
    WHERE a.category IN (SELECT category FROM user_categories)
    AND a.user_id != :user_id
    ORDER BY a.rating DESC, a.date DESC
    LIMIT :limit
    """)
    result = await db.execute(query, {
        "user_id": user_id,
        "limit": limit
    })
    return result.all()

# Операции с подписчиками
async def follow_user(db: AsyncSession, follower_id: int, followed_id: int):
    query = text("""
    INSERT INTO followers (follower_id, followed_id)
    VALUES (:follower_id, :followed_id)
    ON CONFLICT DO NOTHING
    RETURNING *
    """)
    try:
        result = await db.execute(query, {
            "follower_id": follower_id,
            "followed_id": followed_id
        })
        await db.commit()
        return result.first()
    except Exception as e:
        await db.rollback()
        raise e

async def unfollow_user(db: AsyncSession, follower_id: int, followed_id: int):
    query = text("""
    DELETE FROM followers
    WHERE follower_id = :follower_id AND followed_id = :followed_id
    """)
    try:
        await db.execute(query, {
            "follower_id": follower_id,
            "followed_id": followed_id
        })
        await db.commit()
        return True
    except Exception as e:
        await db.rollback()
        raise e 