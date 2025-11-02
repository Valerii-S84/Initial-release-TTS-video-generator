from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .db import get_db, Base, engine
from .core.config import settings
from .models import User
from .core.logging import bind_user
import sentry_sdk


# Ensure tables exist (idempotent)
Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security = HTTPBearer(auto_error=True)


class AuthService:
    SECRET_KEY = settings.JWT_SECRET_KEY or os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
    ALGORITHM = "HS256"
    ACCESS_EXPIRES_MIN = int(os.getenv("JWT_ACCESS_MIN", "60"))  # 60 min
    REFRESH_EXPIRES_DAYS = int(os.getenv("JWT_REFRESH_DAYS", "14"))

    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return pwd_context.verify(password, hashed)

    @staticmethod
    def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=AuthService.ACCESS_EXPIRES_MIN))
        payload = {"sub": user_id, "type": "access", "exp": expire}
        return jwt.encode(payload, AuthService.SECRET_KEY, algorithm=AuthService.ALGORITHM)

    @staticmethod
    def create_refresh_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=AuthService.REFRESH_EXPIRES_DAYS))
        payload = {"sub": user_id, "type": "refresh", "exp": expire}
        return jwt.encode(payload, AuthService.SECRET_KEY, algorithm=AuthService.ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> dict:
        return jwt.decode(token, AuthService.SECRET_KEY, algorithms=[AuthService.ALGORITHM])

    @staticmethod
    async def get_current_user(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db),
    ) -> User:
        if not credentials or not credentials.credentials:
            raise HTTPException(status_code=401, detail="Missing credentials")
        token = credentials.credentials
        try:
            payload = AuthService.decode_token(token)
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Invalid token type")
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token payload")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        # Expose user-based rate limit key for middleware and logging
        try:
            request.state.rate_key = f"user:{user.id}"
        except Exception:
            pass
        try:
            bind_user(user.id)
        except Exception:
            pass
        try:
            sentry_sdk.set_user({"id": str(user.id), "email": user.email})
        except Exception:
            pass
        return user


# Pydantic schemas kept inline to avoid extra files
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token expiry in seconds")


from fastapi import APIRouter

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenPair)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=data.email, hashed_password=AuthService.hash_password(data.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    access = AuthService.create_access_token(str(user.id))
    refresh = AuthService.create_refresh_token(str(user.id))
    return TokenPair(access_token=access, refresh_token=refresh, expires_in=AuthService.ACCESS_EXPIRES_MIN * 60)


@router.post("/login", response_model=TokenPair)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not AuthService.verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access = AuthService.create_access_token(str(user.id))
    refresh = AuthService.create_refresh_token(str(user.id))
    return TokenPair(access_token=access, refresh_token=refresh, expires_in=AuthService.ACCESS_EXPIRES_MIN * 60)


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=TokenPair)
def refresh_tokens(payload: RefreshRequest, db: Session = Depends(get_db)):
    try:
        decoded = AuthService.decode_token(payload.refresh_token)
        if decoded.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        uid = decoded.get("sub")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == int(uid)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    access = AuthService.create_access_token(str(user.id))
    refresh = AuthService.create_refresh_token(str(user.id))
    return TokenPair(access_token=access, refresh_token=refresh, expires_in=AuthService.ACCESS_EXPIRES_MIN * 60)
