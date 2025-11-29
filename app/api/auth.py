from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from app.core import config, security
from app.models.token import Token
from app.models.user import UserLogin
from app.core.logging_config import security_logger

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    
    if (user_data.username == config.settings.ADMIN_USERNAME and 
        user_data.password == config.settings.ADMIN_PASSWORD):
        
        security_logger.info(f"Login Successful for user: {user_data.username}")
        
        access_token = security.create_access_token(data={"sub": user_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    
    security_logger.warning(f"Login Failed for username: {user_data.username}")
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )