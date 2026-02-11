from fastapi import FastAPI
import uvicorn

from lifespan import lifespan
from router import register_routers

app = FastAPI(lifespan=lifespan)
register_routers(app)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)