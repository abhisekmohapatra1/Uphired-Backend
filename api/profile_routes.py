import os
import aiofiles
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from db.database import get_db
from db.models import User, UserProfile
from api.auth_routes import get_current_user
from agents.resume_parser_agent import parse_resume
from loguru import logger

router = APIRouter(prefix="/profile", tags=["profile"])

UPLOAD_DIR = "./uploads/resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload PDF resume → parse with LLM → save profile to DB.
    Returns the extracted profile so user can review/edit it.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    # Save PDF to disk
    pdf_path = f"{UPLOAD_DIR}/{current_user.id}_{file.filename}"
    async with aiofiles.open(pdf_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    logger.info(f"Resume saved: {pdf_path}")

    # Parse resume with LLM
    try:
        parsed = await parse_resume(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Resume parsing failed: {e}")

    # Save or update profile in DB
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()

    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)

    # Map parsed fields to profile model
    for field, value in parsed.items():
        if hasattr(profile, field) and value:
            setattr(profile, field, value)

    profile.resume_path = pdf_path
    await db.commit()
    await db.refresh(profile)

    return {"message": "Resume parsed successfully", "profile": parsed}


@router.get("/me")
async def get_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found. Upload your resume first.")
    return profile


class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None


@router.patch("/me")
async def update_profile(
    req: ProfileUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(UserProfile).where(UserProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    for field, value in req.model_dump(exclude_none=True).items():
        setattr(profile, field, value)

    await db.commit()
    return {"message": "Profile updated"}