from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, Depends

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    yield
    print("Shutting down")