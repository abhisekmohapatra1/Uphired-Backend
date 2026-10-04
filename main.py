from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from api.routes import router
from api.websocket import ws_router
from api.auth_routes import router as auth_router
from api.profile_routes import router as profile_router
from db.database import init_db
from evals.trace import init_tracing
import os

os.makedirs("./uploads/resumes", exist_ok=True)
os.makedirs("./uploads/screenshots", exist_ok=True)

app = FastAPI(title="Job Hunt Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router,         prefix="/api")
app.include_router(ws_router)
app.include_router(auth_router,    prefix="/api")
app.include_router(profile_router, prefix="/api")

# Serve uploaded screenshots so frontend can display them
app.mount("/uploads", StaticFiles(directory="./uploads"), name="uploads")
    

@app.on_event("startup")
async def startup():
    await init_db()
    init_tracing()
    logger.info("Job Hunt Agent API v2 started")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}