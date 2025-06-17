from fastapi import FastAPI, HTTPException, Depends, status, Request, Response, Form, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import database
from sqlalchemy import select, text
import os
import aiohttp
import json
from openai import OpenAI
import PyPDF2
import tempfile

photo_AI = ""
deepseek_AI = ""

# Security configuration
SECRET_KEY = ""  # In production, use a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware для установки заголовков ответа
@app.middleware("http")
async def add_json_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Content-Type"] = "application/json"
    return response

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

# Создаем директорию для статей, если она не существует
ARTICLES_DIR = "static/articles"
os.makedirs(ARTICLES_DIR, exist_ok=True)

# Создаем директорию для загрузок, если она не существует
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Монтируем статические файлы для загруженных файлов
app.mount("/static/articles", StaticFiles(directory="static/articles"), name="articles")

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    login: Optional[str] = None

class UserBase(BaseModel):
    login: str
    name: str
    surname: str
    about: Optional[str] = None
    type: Optional[str] = 'user'

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    rating: float
    like: Optional[str] = None
    role: Optional[str] = None  # Добавляем поле role
    
    class Config:
        from_attributes = True

class Article(BaseModel):
    id: int
    title: str
    preview_image: Optional[str] = None
    authors: str
    abstract: Optional[str] = None
    keywords: Optional[str] = None
    rating: Optional[float] = None
    file_path: str
    category: str
    date: str

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    name: Optional[str] = None
    surname: Optional[str] = None
    login: Optional[str] = None
    about: Optional[str] = None

# Модели для ответов API
class UserStats(BaseModel):
    articles_count: int
    favorites_count: int
    followers_count: int
    following_count: int
    average_rating: float

class UserProfileData(BaseModel):
    user: User
    stats: UserStats
    is_following: bool = False

# Helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(database.get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        login: str = payload.get("sub")
        if login is None:
            raise credentials_exception
        token_data = TokenData(login=login)
    except JWTError:
        raise credentials_exception
    user = await database.get_user_by_login(db, login=token_data.login)
    if user is None:
        raise credentials_exception
    print("get_current_user returned:", user)  # Добавляем лог
    return user

# API routes
@app.post("/api/token")
async def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(database.get_db)
):
    try:
        print(f"Login attempt for user: {form_data.username}")
        
        user = await database.get_user_by_login(db, form_data.username)
        if not user:
            print("User not found")
            response.headers["Content-Type"] = "application/json"
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Неверный логин или пароль"}
            )
            
        if not verify_password(form_data.password, user["password"]):
            print("Invalid password")
            response.headers["Content-Type"] = "application/json"
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Неверный логин или пароль"}
            )
            
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["login"]}, expires_delta=access_token_expires
        )
        
        print("Login successful")
        response.headers["Content-Type"] = "application/json"
        return {"access_token": access_token, "token_type": "bearer", "user_id": user["id"]}
        
    except Exception as e:
        print(f"Login error: {str(e)}")
        response.headers["Content-Type"] = "application/json"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Произошла ошибка при входе"}
        )

@app.get("/api/users/me", response_model=User)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    print("read_users_me received user:", current_user)
    print("User role:", current_user.get('role'))
    return current_user

@app.post("/api/register", response_model=User)
async def register(user: UserCreate, db: AsyncSession = Depends(database.get_db)):
    try:
        # Проверяем, существует ли пользователь
        db_user = await database.get_user_by_login(db, user.login)
        if db_user:
            raise HTTPException(
                status_code=400,
                detail="Пользователь с таким логином уже существует"
            )
        
        # Хешируем пароль
        hashed_password = get_password_hash(user.password)
        
        # Создаем пользователя
        db_user = await database.create_user(
            db,
            login=user.login,
            password=hashed_password,
            name=user.name,
            surname=user.surname
        )
        if not db_user:
            raise HTTPException(
                status_code=500,
                detail="Ошибка при создании пользователя"
            )
        return db_user
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/api/users/{user_id}/articles")
async def get_user_articles(
    user_id: int,
    db: AsyncSession = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    """Получение статей пользователя"""
    try:
        query = text("""
            SELECT id, title, preview_image, authors, abstract, keywords, rating, file_path, category, date
            FROM articles
            WHERE user_id = :user_id
            ORDER BY date DESC
        """)
        result = await db.execute(query, {"user_id": user_id})
        articles = result.fetchall()
        
        if not articles:
            return []
            
        return [{
            "id": article[0],
            "title": article[1],
            "preview_image": article[2],
            "authors": article[3],
            "abstract": article[4],
            "keywords": article[5],
            "rating": article[6],
            "file_path": article[7],
            "category": article[8],
            "date": article[9]
        } for article in articles]
    except Exception as e:
        print(f"Error getting user articles: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при получении статей пользователя")

@app.get("/api/users/{user_id}", response_model=User)
async def get_user(user_id: int, db: AsyncSession = Depends(database.get_db)):
    user = await database.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/api/articles/top")
async def get_top_articles(
    db: AsyncSession = Depends(database.get_db)
):
    """Получение 3 случайных статей"""
    try:
        query = text("""
            SELECT id, title, preview_image, authors, abstract, keywords, rating, file_path, category, date
            FROM articles
            ORDER BY RANDOM()
            LIMIT 3
        """)
        
        result = await db.execute(query)
        articles = result.all()
        
        formatted_articles = []
        for article in articles:
            formatted_article = {
                "id": article[0],
                "title": article[1],
                "preview_image": article[2],
                "authors": article[3],
                "abstract": article[4],
                "keywords": article[5],
                "rating": article[6],
                "file_path": article[7],
                "category": article[8],
                "date": article[9]
            }
            formatted_articles.append(formatted_article)
        
        return formatted_articles
        
    except Exception as e:
        print(f"Ошибка получения случайных статей: {str(e)}")
        return []

@app.get("/api/articles/{article_id}", response_model=Article)
async def get_article(article_id: int, db: AsyncSession = Depends(database.get_db)):
    article = await database.get_article_by_id(db, article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

@app.get("/api/search/articles")
async def search_articles(
    request: Request,
    db: AsyncSession = Depends(database.get_db)
):
    try:
        # Получаем параметры из URL
        query_params = request.query_params
        search_query = query_params.get("title", "")
        sort_by = query_params.get("sort_by", "date")
        sort_order = query_params.get("sort_order", "desc")
        category = query_params.get("category")
        min_rating = query_params.get("min_rating")
        year = query_params.get("year")
        
        print(f"Поисковый запрос: {search_query}")
        print(f"Параметры сортировки: {sort_by} {sort_order}")
        print(f"Фильтры: категория={category}, рейтинг={min_rating}, год={year}")
        
        # Проверяем, есть ли хотя бы один фильтр или поисковый запрос
        has_filters = any([category, min_rating, year])
        has_search = bool(search_query.strip())
        
        if not has_search and not has_filters:
            # Если нет ни поиска, ни фильтров, возвращаем все статьи
            base_query = """
                SELECT id, title, preview_image, authors, abstract, keywords, rating, file_path, category, date
                FROM articles
            """
            params = {}
        else:
            # Формируем базовый запрос
            conditions = []
            params = {}
            
            # Добавляем условие поиска по названию, если есть
            if has_search:
                conditions.append("LOWER(title) LIKE LOWER(:title)")
                params["title"] = f"%{search_query}%"
            
            # Добавляем фильтры
            if category:
                conditions.append("category = :category")
                params["category"] = category
                
            if min_rating:
                conditions.append("rating >= :min_rating")
                params["min_rating"] = float(min_rating)
                
            if year:
                conditions.append("strftime('%Y', date) = :year")
                params["year"] = year
            
            # Формируем WHERE часть
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            base_query = f"""
                SELECT id, title, preview_image, authors, abstract, keywords, rating, file_path, category, date
                FROM articles 
                WHERE {where_clause}
            """
        
        # Добавляем сортировку
        valid_sort_fields = ["date", "rating", "title"]
        valid_sort_orders = ["asc", "desc"]
        
        if sort_by not in valid_sort_fields:
            sort_by = "date"
        if sort_order not in valid_sort_orders:
            sort_order = "desc"
            
        query = text(base_query + f" ORDER BY {sort_by} {sort_order}")
        
        print(f"SQL запрос: {query}")
        print(f"Параметры: {params}")
        
        result = await db.execute(query, params)
        articles = result.all()
        
        print(f"Найдено статей: {len(articles)}")
        
        formatted_articles = []
        for article in articles:
            formatted_article = {
                "id": article[0],
                "title": article[1],
                "preview_image": article[2],
                "authors": article[3],
                "abstract": article[4],
                "keywords": article[5],
                "rating": article[6],
                "file_path": article[7],
                "category": article[8],
                "date": article[9]
            }
            formatted_articles.append(formatted_article)
        
        return formatted_articles
        
    except Exception as e:
        print(f"Ошибка поиска: {str(e)}")
        return []

@app.put("/api/users/me/update", response_model=User)
async def update_user_data(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(database.get_db)
):
    updated_user = await database.update_user(
        db,
        current_user.id,
        user_data.dict(exclude_unset=True)
    )
    if updated_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user

# Эндпоинты для профиля пользователя
@app.get("/api/users/{user_id}/profile")
async def get_user_profile(
    user_id: int,
    db: AsyncSession = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    """Получение профиля пользователя"""
    try:
        # Получаем данные пользователя
        query = text("""
            SELECT id, name, surname, login, rating, type, about, like
            FROM users
            WHERE id = :user_id
        """)
        result = await db.execute(query, {"user_id": user_id})
        user_data = result.fetchone()
        
        if not user_data:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        
        # Получаем статистику пользователя
        stats = await database.get_user_stats(db, user_id)
        
        return {
            "user": {
                "id": user_data[0],
                "name": user_data[1],
                "surname": user_data[2],
                "login": user_data[3],
                "rating": user_data[4],
                "role": user_data[5],  # type из базы данных теперь будет role
                "about": user_data[6],
                "like": user_data[7]
            },
            "stats": stats
        }
    except Exception as e:
        print(f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при получении профиля пользователя")

@app.get("/api/users/{user_id}/favorites", response_model=List[Article])
async def get_user_favorites(
    user_id: int,
    db: AsyncSession = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return await database.get_user_favorites(db, user_id)

@app.post("/api/articles/{article_id}/favorite")
async def add_to_favorites(
    article_id: int,
    db: AsyncSession = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        await database.add_to_favorites(db, current_user["id"], article_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/users/{user_id}/favorites/{article_id}")
async def remove_from_favorites(user_id: int, article_id: int, db: AsyncSession = Depends(database.get_db)):
    try:
        # Получаем пользователя
        user = await database.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
            
        # Проверяем, есть ли статья в избранном
        if article_id not in user.like:
            raise HTTPException(status_code=400, detail="Статья не найдена в избранном")
            
        # Удаляем статью из избранного
        user.like.remove(article_id)
        await db.commit()
        
        return {"message": "Статья удалена из избранного"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/recommended", response_model=List[Article])
async def get_recommended_articles(
    user_id: int,
    limit: int = 10,
    db: AsyncSession = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return await database.get_recommended_articles(db, user_id, limit)

@app.post("/api/users/{user_id}/follow")
async def follow_user(
    user_id: int,
    db: AsyncSession = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user["id"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")
    try:
        await database.follow_user(db, current_user["id"], user_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/users/{user_id}/follow")
async def unfollow_user(
    user_id: int,
    db: AsyncSession = Depends(database.get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        await database.unfollow_user(db, current_user["id"], user_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Static files routes
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/index.html")
async def index():
    return FileResponse("static/index.html")

@app.get("/my_profile")
async def profile():
    return FileResponse("static/my_profile.html")

@app.get("/my_profile.html")
async def profile_html():
    return FileResponse("static/my_profile.html")

@app.get("/404")
async def not_found():
    return RedirectResponse(url="/static/404.html")

@app.get("/admin.html")
async def admin_panel():
    return FileResponse("static/admin.html")

# Admin endpoints
async def check_admin_access(current_user: dict = Depends(get_current_user)):
    print(f"Checking admin access for user: {current_user}")
    print(f"User role: {current_user.get('role')}")
    if current_user.get('role') != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

@app.get("/api/admin/dashboard")
async def admin_dashboard(
    db: AsyncSession = Depends(database.get_db),
    admin: dict = Depends(check_admin_access)
):
    try:
        # Get dashboard statistics
        users_count = await db.execute(text("SELECT COUNT(*) FROM users"))
        articles_count = await db.execute(text("SELECT COUNT(*) FROM articles"))
        new_users = await db.execute(text("SELECT COUNT(*) FROM users WHERE id > (SELECT MAX(id) - 10 FROM users)"))
        active_sessions = await db.execute(text("SELECT COUNT(DISTINCT user_id) FROM favorites"))

        return {
            "total_users": users_count.scalar() or 0,
            "total_articles": articles_count.scalar() or 0,
            "new_users": new_users.scalar() or 0,
            "active_sessions": active_sessions.scalar() or 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/admin/users")
async def admin_get_users(
    db: AsyncSession = Depends(database.get_db),
    admin: dict = Depends(check_admin_access)
):
    try:
        result = await db.execute(text("SELECT * FROM users"))
        users = result.all()
        return [
            {
                "id": user[0],
                "name": user[1],
                "surname": user[2],
                "email": user[3],
                "type": user[9] if len(user) > 9 else "user",
                "status": "active"
            }
            for user in users
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/admin/articles")
async def admin_get_articles(
    db: AsyncSession = Depends(database.get_db),
    admin: dict = Depends(check_admin_access)
):
    try:
        result = await db.execute(text("""
            SELECT a.*, u.name, u.surname 
            FROM articles a 
            LEFT JOIN users u ON a.user_id = u.id
        """))
        articles = result.all()
        return [
            {
                "id": article[0],
                "title": article[1],
                "author": f"{article[11]} {article[12]}" if article[11] and article[12] else "Unknown",
                "category": article[8],
                "rating": float(article[6]) if article[6] is not None else 0.0,
                "status": "published"
            }
            for article in articles
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/admin/users")
async def admin_create_user(
    user_data: dict,
    db: AsyncSession = Depends(database.get_db),
    admin: dict = Depends(check_admin_access)
):
    try:
        hashed_password = get_password_hash(user_data["password"])
        new_user = await database.create_user(
            db,
            email=user_data["email"],
            password=hashed_password,
            name=user_data["name"],
            surname=user_data["surname"],
            type=user_data.get("type", "user")
        )
        return new_user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(
    user_id: int,
    db: AsyncSession = Depends(database.get_db),
    admin: dict = Depends(check_admin_access)
):
    try:
        await db.execute(
            text("DELETE FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        )
        await db.commit()
        return {"status": "success"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/api/admin/articles/{article_id}")
async def admin_delete_article(
    article_id: int,
    db: AsyncSession = Depends(database.get_db),
    admin: dict = Depends(check_admin_access)
):
    try:
        await db.execute(
            text("DELETE FROM articles WHERE id = :article_id"),
            {"article_id": article_id}
        )
        await db.commit()
        return {"status": "success"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/articles")
async def create_article(
    title: str = Form(...),
    abstract: str = Form(...),
    authors: str = Form(...),
    keywords: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
    preview_url: str = Form(None),
    db: AsyncSession = Depends(database.get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Сначала создаем запись в БД, чтобы получить ID
        query = text("""
            INSERT INTO articles (title, preview_image, abstract, keywords, rating, file_path, user_id, date, authors, category)
            VALUES (:title, :preview_image, :abstract, :keywords, :rating, :file_path, :user_id, :date, :authors, :category)
            RETURNING id
        """)
        
        # Выполняем запрос с временным путем к файлу
        result = await db.execute(
            query,
            {
                "title": title,
                "preview_image": preview_url,
                "abstract": abstract,
                "keywords": keywords,
                "rating": 0.0,
                "file_path": "temp",  # Временный путь
                "user_id": current_user["id"],
                "date": datetime.now().strftime("%Y-%m-%d"),
                "authors": authors,
                "category": category
            }
        )
        
        await db.commit()
        article_id = result.scalar()
        
        # Сохраняем файл с ID статьи в качестве имени
        file_path = f"articles/{article_id}.pdf"
        full_path = os.path.join("static", file_path)
        
        with open(full_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Обновляем путь к файлу в БД
        await db.execute(
            text("UPDATE articles SET file_path = :file_path WHERE id = :id"),
            {"file_path": file_path, "id": article_id}
        )
        await db.commit()
        
        return {"id": article_id, "message": "Статья успешно создана"}
        
    except Exception as e:
        await db.rollback()
        print(f"Ошибка при создании статьи: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при создании статьи")

@app.get("/api/articles/{article_id}/file")
async def get_article_file(article_id: int, db: AsyncSession = Depends(database.get_db)):
    try:
        # Получаем путь к файлу из БД
        result = await db.execute(
            text("SELECT file_path FROM articles WHERE id = :id"),
            {"id": article_id}
        )
        file_path = result.scalar_one_or_none()
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Статья не найдена")
            
        full_path = os.path.join("static", file_path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Файл статьи не найден")
            
        return FileResponse(full_path)
        
    except Exception as e:
        print(f"Ошибка при получении файла статьи: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при получении файла статьи")

# Функция для извлечения текста из PDF
def extract_text_from_pdf(pdf_file):
    try:
        # Создаем временный файл для сохранения загруженного PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name

        # Открываем PDF и извлекаем текст
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

        # Удаляем временный файл
        os.unlink(temp_file_path)
        
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

# Функция для генерации данных статьи с помощью ИИ
async def generate_article_data(pdf_text: str):
    try:
        client = OpenAI(
            api_key= deepseek_AI,
            base_url="https://api.deepseek.com"
        )

        # Формируем промпт для анализа статьи
        prompt = f"""Проанализируй текст научной статьи и верни данные в формате JSON:
        {{
            "title": "название статьи",
            "authors": "авторы через запятую",
            "tags": "ключевые слова через запятую",
            "abstract": "краткое описание статьи"
        }}

        Текст статьи:
        {pdf_text}

        Верни только JSON без дополнительного текста."""

        # Отправляем запрос к ИИ (без await, так как это синхронный метод)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты - ассистент для анализа научных статей. Твоя задача - извлечь основную информацию из текста и вернуть её в формате JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        # Получаем ответ
        if not response.choices or not response.choices[0].message.content:
            raise Exception("Empty response from AI")

        # Извлекаем JSON из ответа
        content = response.choices[0].message.content.strip()
        
        # Находим JSON в тексте (на случай, если ИИ добавил дополнительный текст)
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            raise Exception("No JSON found in response")
            
        json_str = content[json_start:json_end]
        
        try:
            # Парсим JSON
            data = json.loads(json_str)
            
            # Проверяем наличие всех необходимых полей
            required_fields = ['title', 'authors', 'tags', 'abstract']
            for field in required_fields:
                if field not in data:
                    data[field] = ""
                    
            return data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw content: {content}")
            raise Exception(f"Failed to parse JSON response: {str(e)}")

    except Exception as e:
        print(f"Error in generate_article_data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate article data: {str(e)}"
        )

@app.post("/api/extract_data")
async def extract_data(file: UploadFile = File(...)):
    try:
        # Проверяем тип файла
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Извлекаем текст из PDF
        pdf_text = extract_text_from_pdf(file.file)
        if not pdf_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Ограничиваем размер текста для экономии токенов
        pdf_text = pdf_text[:4000]

        # Генерируем данные с помощью ИИ
        try:
            article_data = await generate_article_data(pdf_text)
            return article_data
        except Exception as e:
            print(f"Error generating article data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate article data: {str(e)}")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        # Убеждаемся, что файл закрыт
        await file.close()

@app.delete("/api/articles/{article_id}")
async def delete_article(article_id: int, db: AsyncSession = Depends(database.get_db)):
    try:
        # Получаем статью
        stmt = select(Article).where(Article.id == article_id)
        result = await db.execute(stmt)
        article = result.scalar_one_or_none()
        
        if not article:
            raise HTTPException(status_code=404, detail="Статья не найдена")
            
        # Удаляем файл статьи
        file_path = os.path.join("static", article.file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Удаляем запись из базы данных
        await db.execute(text("DELETE FROM articles WHERE id = :id"), {"id": article_id})
        await db.commit()
        
        return {"message": "Статья успешно удалена"}
        
    except Exception as e:
        await db.rollback()
        print(f"Ошибка при удалении статьи: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_preview")
async def generate_preview(
    title: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(database.get_db)
):
    try:
        # Формируем промпт для генерации изображения
        prompt = f"Создай превью для статьи: {title}"
        
        # Генерируем изображение
        image_url = await free_generate(prompt, photo_AI) 
        
        if not image_url:
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать изображение")
            
        return {"preview_url": image_url}
        
    except Exception as e:
        print(f"Error generating preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def free_generate(prompt: str, api_key: str):
    async with aiohttp.ClientSession() as session:
        payload = {
            "token": api_key,
            "prompt": prompt,
            "stream": True
        }
        
        async with session.post(
            "https://neuroimg.art/api/v1/free-generate",
            json=payload
        ) as response:
            async for line in response.content:
                if line:
                    data = json.loads(line)
                    if data["status"] == "SUCCESS":
                        return data["image_url"]
                    print(f"Статус: {data['status']}")

# Инициализация клиента OpenAI
client = OpenAI(
    api_key= deepseek_AI,
    base_url="https://api.deepseek.com"
)

@app.get("/api/search/ai")
async def ai_search_articles(
    request: Request,
    db: AsyncSession = Depends(database.get_db)
):
    try:
        # Получаем поисковый запрос
        query_params = request.query_params
        search_query = query_params.get("query", "")
        
        if not search_query:
            return []
            
        # Получаем все статьи из БД
        result = await db.execute(text("""
            SELECT id, title, preview_image, authors, abstract, keywords, rating, file_path, category, date
            FROM articles
        """))
        all_articles = result.all()
        
        # Формируем список названий статей для ИИ
        article_titles = [article[1] for article in all_articles]
        
        # Формируем промпт для ИИ
        prompt = f"""
        У меня есть список названий научных статей. Пользователь ищет: "{search_query}"
        
        Список статей:
        {chr(10).join([f"{i+1}. {title}" for i, title in enumerate(article_titles)])}
        
        Пожалуйста, проанализируй поисковый запрос и верни номера статей, которые наиболее релевантны запросу.
        Верни только номера статей через запятую, без дополнительного текста.
        Например: 1, 3, 5
        """
        
        # Получаем ответ от ИИ
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that helps find relevant scientific articles."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        # Получаем номера статей из ответа ИИ
        ai_response = response.choices[0].message.content.strip()
        relevant_indices = [int(idx.strip()) - 1 for idx in ai_response.split(',')]
        
        # Формируем результат
        formatted_articles = []
        for idx in relevant_indices:
            if 0 <= idx < len(all_articles):
                article = all_articles[idx]
                formatted_article = {
                    "id": article[0],
                    "title": article[1],
                    "preview_image": article[2],
                    "authors": article[3],
                    "abstract": article[4],
                    "keywords": article[5],
                    "rating": article[6],
                    "file_path": article[7],
                    "category": article[8],
                    "date": article[9]
                }
                formatted_articles.append(formatted_article)
        
        return formatted_articles
        
    except Exception as e:
        print(f"Ошибка ИИ-поиска: {str(e)}")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 