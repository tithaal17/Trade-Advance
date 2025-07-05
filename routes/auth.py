from fastapi import APIRouter, Request, Query, HTTPException
from fastapi import  Request, Form, Depends
import bcrypt,datetime
from fastapi.responses import RedirectResponse
import sqlite3, config
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter()

def get_db_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@router.get("/login")
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/signup")
def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@router.post("/signup")
def signup(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    connection = get_db_connection()
    cursor = connection.cursor()

    # Check if email already exists
    cursor.execute("SELECT * FROM user WHERE email = ?", (email,))
    existing_user = cursor.fetchone()

    if existing_user:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Email is already registered, please login"})


    # Hash the password before storing
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    cursor.execute("INSERT INTO user (username, email, password_hash) VALUES (?, ?, ?)",(username, email, hashed_password.decode("utf-8")))

    connection.commit()
    connection.close()

    return RedirectResponse(url="/index", status_code=303)

@router.post("/login")
def login(request: Request,email: str = Form(...), password: str = Form(...)):
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT id, username, password_hash FROM user WHERE email = ?", (email,))
    user = cursor.fetchone()

    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid Email"})

    user_id, username, stored_password_hash = user

    # Verify password
    if not bcrypt.checkpw(password.encode("utf-8"), stored_password_hash.encode("utf-8")):
        return templates.TemplateResponse("login.html", {"request": request,"error": "Invalid Password"})
    
    request.session["user_id"] = user["id"]
    
    connection.close()
    
    return RedirectResponse(url="/index", status_code=303)

@router.get("/get_current_user")
def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {"user_id": user_id}