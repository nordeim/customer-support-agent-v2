# Project Architecture Document

## 1. Executive Summary

This document provides a comprehensive overview of the architecture, design, and implementation of the Customer Support AI Agent. It is intended to be the definitive source of truth for the current state of the codebase, reflecting its evolution from the initial concepts outlined in `AGENT.md` into a mature, production-ready application.

The system is a microservices-based application with a React frontend and a FastAPI backend. At its core is a custom-built AI agent orchestrator that leverages a suite of tools for Retrieval-Augmented Generation (RAG), conversation memory, and more, to provide intelligent and context-aware customer support.

**Project Status:** Implemented, Functional, and Ready for Future Development.

## 2. System Architecture

The application is designed with a clear separation of concerns between the frontend, backend, and the various services that support them.

### 2.1. High-Level Architecture

-   **Frontend:** A single-page application (SPA) built with **React** and **TypeScript**. It provides the user interface for the chat and handles real-time communication with the backend via WebSockets.
-   **Backend:** A robust API built with **FastAPI**. It serves as the central hub, handling user requests, managing the AI agent, and interfacing with the database and cache.
-   **AI Core:** A **custom agent orchestrator** (`CustomerSupportAgent`) that manages the logic flow. It does **not** use an external framework like Microsoft Agent Framework. It integrates multiple tools to process user input and generate intelligent responses.
-   **Database:** **SQLAlchemy** is used as the ORM. The system is configured to use **SQLite** for development and is ready for **PostgreSQL** in production. **Alembic** is integrated for handling database migrations.
-   **Cache:** **Redis** is used for caching session data and other frequently accessed information to improve performance.
-   **Vector Store:** **ChromaDB** is used as the vector store for the RAG system, enabling efficient similarity searches on document embeddings.

### 2.2. User Interaction Logic Flow

The following diagram illustrates the flow of a user message through the system from the frontend to the backend and back.

```mermaid
graph TD
    subgraph Frontend
        A[User sends message via Chat Interface]
    end

    subgraph Backend
        B[FastAPI Backend]
        C[API Route: /chat]
        D[CustomerSupportAgent.process_message]
        E[1. Load Context from Memory]
        F[2. Process File Attachments]
        G[3. Search Knowledge Base (RAG)]
        H[4. Check for Escalation]
        I[5. Generate Response]
        J[6. Store New Memories]
        K[AgentResponse Object]
    end

    subgraph Services
        L[MemoryTool (SQLite/Postgres)]
        M[AttachmentTool (File System)]
        N[RAGTool (ChromaDB)]
        O[EscalationTool]
    end

    A --> B;
    B --> C;
    C --> D;
    D --> E;
    E --> L;
    D --> F;
    F --> M;
    M --> N;
    D --> G;
    G --> N;
    D --> H;
    H --> O;
    D --> I;
    I --> J;
    J --> L;
    I --> K;
    K --> C;
    C --> B;
    B -- WebSocket/HTTP --> A;
```

## 3. File Hierarchy and Key Components

This section details the project's file structure and the purpose of key files.

```
/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application entry point. Manages startup/shutdown, middleware, and routing.
│   │   ├── config.py             # Production-grade Pydantic settings management.
│   │   ├── database.py           # SQLAlchemy setup and database session management.
│   │   ├── agents/
│   │   │   └── chat_agent.py     # Core of the AI. The custom agent orchestrator that processes messages and uses tools.
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── chat.py       # API endpoints for handling chat messages.
│   │   │   │   └── ...
│   │   │   └── websocket.py      # Logic for the real-time WebSocket communication.
│   │   ├── models/               # SQLAlchemy ORM models (Session, Message, Memory).
│   │   ├── services/             # Services for business logic (e.g., caching, embeddings).
│   │   └── tools/
│   │       ├── rag_tool.py       # Retrieval-Augmented Generation tool, interfaces with ChromaDB.
│   │       ├── memory_tool.py    # Manages conversation history and context.
│   │       ├── attachment_tool.py# Handles file uploads and content extraction.
│   │       └── escalation_tool.py# Determines if a conversation needs to be escalated to a human.
│   ├── alembic/                  # Alembic database migration scripts.
│   ├── data/                     # Data storage for SQLite and ChromaDB.
│   └── tests/                    # Pytest unit and integration tests.
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Main React application component.
│   │   ├── components/
│   │   │   └── ChatInterface.tsx # The primary UI component for the chat.
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts   # Custom React hook for managing the WebSocket connection.
│   │   └── services/
│   │       └── api.ts            # Functions for making HTTP requests to the backend.
│   └── package.json              # Frontend dependencies and scripts.
├── docker-compose.yml            # Docker Compose configuration for development.
├── docker-compose.prod.yml       # Docker Compose configuration for production.
├── GEMINI.md                     # Up-to-date project overview for the AI coding agent.
└── README.md                     # General project README.
```

## 4. Core Components Deep Dive

### 4.1. The Custom AI Agent (`chat_agent.py`)

This is the brain of the application. It does not rely on an external framework. Its primary responsibilities are:

-   **Orchestration:** It follows a defined sequence of steps to process a user's message: load context, process attachments, search the knowledge base, check for escalation, generate a response, and store memories.
-   **Tool Integration:** It directly calls the various tools (`RAGTool`, `MemoryTool`, etc.) and synthesizes their outputs.
-   **State Management:** It manages the context of each conversation session through a custom `AgentContext` class.
-   **Structured Responses:** It returns a well-defined `AgentResponse` object, ensuring that the API layer receives predictable and structured data.

### 4.2. RAG and Knowledge (`rag_tool.py`)

The RAG system allows the agent to answer questions based on a knowledge base.

-   **Embedding:** Documents are chunked and converted into vector embeddings using SentenceTransformer models.
-   **Storage:** These embeddings are stored in ChromaDB.
-   **Retrieval:** When a user asks a question, the RAG tool embeds the query and performs a similarity search in ChromaDB to find the most relevant document chunks. These chunks are then used to generate an informed response.

### 4.3. Configuration (`config.py`)

The system's configuration is managed by a powerful Pydantic `Settings` class. This provides:

-   **Type Safety:** All settings are strongly typed.
-   **Validation:** Custom validators ensure that settings are correct (e.g., creating directories if they don't exist).
-   **Environment-Awareness:** The configuration adapts based on the environment (development vs. production).
-   **Centralization:** All configuration is in one place, making the system easy to manage.

## 5. Future Development

The current architecture is highly modular and extensible, making it well-suited for future development. New tools can be easily added to the agent, and the service-oriented backend can be expanded with new features. The codebase should be considered the single source of truth, and the patterns established within it should be followed for all new contributions.
