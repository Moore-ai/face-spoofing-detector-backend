from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    content: str

@router.get("/health", response_model=HealthResponse)
async def health_check():
    response = HealthResponse(status="ok", content="Hello, from fastapi")
    return response