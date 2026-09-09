import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from backend.app.database import get_session
from backend.app.models import User, UserRegister, UserLogin, TokenResponse
from backend.app.dependencies import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=TokenResponse)
def register_user(req: UserRegister, session: Session = Depends(get_session)):
    # Check if email is already taken
    existing = session.exec(select(User).where(User.email == req.email)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered."
        )

    user = User(
        email=req.email,
        role=req.role if req.role in ["recruiter", "candidate", "admin"] else "candidate",
        hashed_password=hash_password(req.password)
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    token_data = {"sub": str(user.id), "role": user.role, "email": user.email}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=str(user.id),
        role=user.role,
        email=user.email
    )

@router.post("/login", response_model=TokenResponse)
def login_user(req: UserLogin, session: Session = Depends(get_session)):
    # Check credentials
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    token_data = {"sub": str(user.id), "role": user.role, "email": user.email}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=str(user.id),
        role=user.role,
        email=user.email
    )

@router.post("/refresh")
def refresh_access_token(refresh_token: str, session: Session = Depends(get_session)):
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Invalid token type.")

    user_id_str = payload.get("sub")
    user = session.get(User, uuid.UUID(user_id_str))
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    token_data = {"sub": str(user.id), "role": user.role, "email": user.email}
    new_access_token = create_access_token(token_data)
    return {"access_token": new_access_token, "token_type": "bearer"}

@router.get("/me")
def get_current_user_profile(user: User = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at
    }
