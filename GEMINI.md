# Project: Customer Support AI Agent

## Project Overview

This directory contains an enterprise-grade, AI-powered customer support system. The project is **fully implemented** and has evolved beyond its initial design phase, which was documented in `AGENT.md`. The application features a microservices-based architecture with a React frontend and a FastAPI backend.

The core of the application is a sophisticated, **custom-built AI agent orchestrator** (not Microsoft Agent Framework). This agent is capable of real-time chat, Retrieval-Augmented Generation (RAG) for knowledge retrieval, conversation memory management, and processing of user-uploaded documents.

**Project Status:** Implemented, Functional, and Ready for Feature Development/Bug Fixing.

**Key Technologies:**

*   **Frontend:** React, TypeScript, Vite, Tailwind CSS
*   **Backend:** FastAPI, Python 3.11+, SQLAlchemy, Alembic (for migrations)
*   **AI/ML:** A custom agent orchestrator. Tools leverage SentenceTransformers (e.g., `all-MiniLM-L6-v2` or `google/embedding-gemma-256m-it`) and ChromaDB for RAG.
*   **Infrastructure:** Docker, Redis, SQLite (dev), PostgreSQL (prod-ready)

## Architecture

The implemented architecture has positively diverged from the initial plan in `AGENT.md`.

*   **Custom AI Agent:** Instead of the proposed Microsoft Agent Framework, the project uses a more transparent and maintainable custom agent orchestrator located in `backend/app/agents/chat_agent.py`. This orchestrator manages a suite of tools for RAG, memory, and more.
*   **Structured Classes:** The agent logic is built around clear, structured classes like `AgentContext` and `AgentResponse`, ensuring predictable behavior and easier debugging.
*   **Advanced Features:** The system includes production-ready features such as a robust configuration system (`backend/app/config.py`), database migrations with Alembic, and a comprehensive suite of middleware for error handling, rate limiting, and request timing.

## Building and Running

The project is fully runnable. The following instructions have been updated to reflect the most stable and reliable startup procedure.

1.  **Initialize the Python backend:**
    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Initialize the Node.js frontend:**
    ```bash
    cd frontend
    npm install
    ```

**Running the application:**

*   **Using Docker Compose (Recommended):**
    ```bash
    docker-compose up -d
    ```
*   **Manual Development Setup (Verified):**
    *   **Backend:** The startup process has been simplified into a single script. This is the recommended way to run the backend manually.
        ```bash
        ./backend_start.sh
        ```
    *   **Frontend:**
        ```bash
        cd frontend
        npm run dev
        ```
    *   **Note:** The frontend development server uses a proxy to communicate with the backend. Ensure that API calls in the code use relative paths (e.g., `/api/sessions`) so they are correctly routed.

## Recent Stability Fixes

The project recently underwent significant stabilization to resolve a series of build and runtime errors. These fixes have made the development environment more robust.

*   **Build System:** Resolved over 40 TypeScript errors and a critical Vite/Rollup dependency conflict related to `react-syntax-highlighter`. This involved creating `tsconfig.node.json`, installing missing type definitions, and reverting the syntax highlighting packages to a known stable version.
*   **Runtime Environment:**
    *   Fixed an infinite loading screen ("spinning") on the frontend. The root cause was an incorrect API URL configuration that bypassed the Vite proxy, leading to silent CORS errors.
    *   Corrected the backend startup script (`backend_start.sh`) to provide a single, reliable command for launching the server, resolving a final `ECONNREFUSED` connection error.

The application is now confirmed to be stable and buildable.

## Development Conventions

The project follows modern development conventions. While `AGENT.md` provided the initial guidelines, the current codebase should be considered the source of truth for all patterns.

*   **Code Style:**
    *   **Python:** PEP 8, extensive type hints, and a focus on readable, modular code.
    *   **TypeScript:** The project is set up with ESLint and Prettier for code quality.
*   **Testing:**
    *   The testing framework is in place using `pytest` for the backend. New features should be accompanied by corresponding unit and integration tests.
*   **Architecture Patterns:** The codebase demonstrates a service-oriented approach with clear separation of concerns between API routes, agent logic, tools, and services. This pattern should be followed for new features.