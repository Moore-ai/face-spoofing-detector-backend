from fastapi import APIRouter
from pydantic import BaseModel
from util.config import settings

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    content: str
    debug_mode: bool = False

@router.get("/health", response_model=HealthResponse)
async def health_check():
    response = HealthResponse(
        status="ok",
        content="Hello, from fastapi",
        debug_mode=settings.DEBUG_MODE,
    )
    return response