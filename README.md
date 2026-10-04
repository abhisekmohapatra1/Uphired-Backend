# Uphired-Backend

A production-ready FastAPI backend powering Uphired - an AI-driven job hunting agent system. This service orchestrates intelligent job discovery, extraction, ranking, and reporting across multiple job boards using a multi-agent architecture.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Design](#system-design)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Graph Workflow](#graph-workflow)
- [Configuration](#configuration)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Authentication](#authentication)
- [File Uploads](#file-uploads)
- [Evaluation Framework](#evaluation-framework)
- [Error Handling & Resilience](#error-handling--resilience)
- [WebSocket Support](#websocket-support)
- [License](#license)

## Overview

Uphired-Backend implements a LangGraph-powered multi-agent system that automates job hunting. It searches across multiple job platforms, scrapes job listings, extracts structured data, enriches job details, performs intelligent matching against user profiles, and generates actionable reports.

### Key Features

- **Multi-Agent Orchestration**: Coordinated workflow using LangGraph with conditional routing and retries
- **Multi-Platform Job Search**: Integration with RemoteOK, LinkedIn, Indeed, Naukri, Wellfound, Y Combinator Jobs
- **Intelligent Job Matching**: Semantic skill matching with expansion, title relevance scoring, and LLM-based evaluation
- **User Authentication & Profiles**: JWT-based auth with bcrypt password hashing, user profile management, and resume parsing support
- **Real-time Updates**: WebSocket endpoints for live workflow progress tracking
- **Resilient LLM Client**: Multi-key failover for OpenRouter with retry logic and cooldown handling
- **Tracing & Evaluation**: LangSmith integration with comprehensive evaluation framework
- **File Management**: Static file serving for resumes, screenshots, and uploads

## Architecture

The system follows a multi-agent architecture orchestrated by LangGraph. Each agent has a specific responsibility and works together in a pipeline with feedback loops for robustness.

```text
+-----------------+    +---------------------------------------------------------+
¦   FastAPI App   ¦    ¦                    LangGraph Workflow                    ¦
¦  (main.py)      ¦?--?¦  Planner ? Browser ? Extractor ? Ranker ? Summary (END)  ¦
+-----------------+    +---------------------------------------------------------+
         ¦                            ¦
         ¦                            ?
    +----?-----+      +---------------------------------------------------+
    ¦   API     ¦      ¦                      Agents                       ¦
    ¦  Routes   ¦      ¦ planner | browser | extractor | ranker | summary ¦
    +-----------+      +---------------------------------------------------+
         ¦                            ¦
         ?                            ?
    +-----------+              +---------------------------------------------+
    ¦ Database   ¦              ¦              Tools & External                ¦
    ¦  (SQLite)  ¦              ¦ BrowserTools, LLM Tools, Scraping, OpenRouter¦
    +-----------+              +---------------------------------------------+
```

## System Design

### Design Principles

1. **Modularity**: Clear separation between agents, tools, API layers, and data models
2. **Resilience**: Built-in retry mechanisms, fallback logic, and graceful error handling
3. **Scalability**: Asynchronous operations throughout (asyncio, aiosqlite)
4. **Extensibility**: Easy to add new job sites, agents, or tools
5. **Observability**: Comprehensive logging with Loguru, tracing via LangSmith

### Workflow Pattern

The job search workflow follows this stateful pattern:
1. **Input**: User query, skills, experience, preferred sites, market (global/india)
2. **Planning**: Generate optimized search queries (2-4 variants) based on intent
3. **Browsing**: Execute searches across selected platforms with site-specific handlers
4. **Extraction**: Enrich scraped jobs with inferred skills and structured data
5. **Ranking**: Filter relevance, compute match scores (semantic + title + description), deduplicate, LLM re-scoring for top results
6. **Summary**: Generate actionable markdown report
7. **Feedback Loops**: Retry on no results, broaden search if still insufficient
