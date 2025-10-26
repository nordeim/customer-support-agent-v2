# Project: Customer Support AI Agent

## Project Overview

This directory contains the blueprint for an enterprise-grade, AI-powered customer support system. The project is currently in the planning and design phase, with the `AGENT.md` file serving as a detailed implementation guide for an AI agent. The final product, as described in `README.md`, will be a microservices-based application with a React frontend and a FastAPI backend.

The core of the application is an intelligent agent built with the Microsoft Agent Framework, capable of real-time chat, Retrieval-Augmented Generation (RAG) for knowledge retrieval, and processing of user-uploaded documents.

**Key Technologies:**

*   **Frontend:** React, TypeScript, Vite, Tailwind CSS
*   **Backend:** FastAPI, Python 3.11+, SQLAlchemy
*   **AI/ML:** Microsoft Agent Framework, Google EmbeddingGemma, ChromaDB
*   **Infrastructure:** Docker, Redis, SQLite (dev), PostgreSQL (prod)

## Building and Running

The project is not yet implemented. To begin the implementation, follow the steps outlined in `AGENT.md`. The primary steps to get started are:

1.  **Set up the directory structure:**
    ```bash
    mkdir -p backend/app/{agents,tools,api/routes,models,services,utils}
    mkdir -p frontend/src/{components,hooks,services,types}
    mkdir -p monitoring scripts
    ```

2.  **Initialize the Python backend:**
    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Initialize the Node.js frontend:**
    ```bash
    cd frontend
    npm install
    ```

**Running the application (once implemented):**

*   **Using Docker Compose (Recommended):**
    ```bash
    docker-compose up -d
    ```
*   **Manual Development Setup:**
    *   **Backend:**
        ```bash
        cd backend
        source venv/bin/activate
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
        ```
    *   **Frontend:**
        ```bash
        cd frontend
        npm run dev
        ```

## Development Conventions

The `AGENT.md` file outlines a clear set of development conventions to be followed during implementation:

*   **Phased Implementation:** The project is to be built in phases, starting with the backend foundation, then the core agent and tools, followed by the API routes and frontend.
*   **Code Style:**
    *   **Python:** PEP 8, type hints, and a maximum line length of 88 characters.
    *   **TypeScript:** ESLint and Prettier are to be used.
*   **Testing:**
    *   A minimum of 80% test coverage is required for all new code.
    *   Both unit and integration tests are expected.
    *   Backend tests are run with `pytest`, and frontend tests with `npm test`.
*   **Architecture Patterns:** The project specifies design patterns to be used for services, tools, and error handling.
*   **Commits:** The `AGENT.md` suggests committing after each working feature is implemented.
