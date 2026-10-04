from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    email         = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)


class UserProfile(Base):
    __tablename__ = "user_profiles"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(Integer, ForeignKey("users.id"), unique=True)

    # Personal info — filled from resume
    full_name       = Column(String(255))
    email           = Column(String(255))
    phone           = Column(String(50))
    address         = Column(String(500))
    city            = Column(String(100))
    state           = Column(String(100))
    country         = Column(String(100))
    pincode         = Column(String(20))
    linkedin_url    = Column(String(500))
    github_url      = Column(String(500))
    portfolio_url   = Column(String(500))

    # Parsed resume data
    resume_path     = Column(String(500))   # path to uploaded PDF
    skills          = Column(JSON)          # ["Python", "FastAPI", ...]
    experience_years = Column(Integer)
    work_history    = Column(JSON)          # [{company, role, duration, description}]
    education       = Column(JSON)          # [{degree, institution, year}]
    certifications  = Column(JSON)          # ["AWS SAA", ...]
    summary         = Column(Text)          # professional summary

    updated_at      = Column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    title         = Column(String(255))
    company       = Column(String(255))
    location      = Column(String(255))
    salary        = Column(String(100))
    skills        = Column(JSON)
    description   = Column(Text)
    url           = Column(String(500), unique=True)
    source        = Column(String(50))
    match_score   = Column(Float, default=0.0)
    created_at    = Column(DateTime, default=datetime.utcnow)


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    id            = Column(String(36), primary_key=True)
    user_query    = Column(Text)
    status        = Column(String(20))
    logs          = Column(JSON)
    result        = Column(JSON)
    created_at    = Column(DateTime, default=datetime.utcnow)