# Project: Customer Support AI Agent

## 1. Project Overview

This is an enterprise-grade, AI-powered customer support system featuring a microservices-based architecture with a React frontend and a FastAPI backend. The project is fully implemented, functional, and has evolved significantly beyond its initial design.

The core of the application is a sophisticated, custom-built AI agent orchestrator capable of real-time chat (via WebSockets), Retrieval-Augmented Generation (RAG) for knowledge retrieval, conversation memory management, and processing of user-uploaded documents.

**Project Status:** Implemented, Functional, and Ready for Feature Development/Bug Fixing.

## 2. Technology Stack

*   **Frontend:** React, TypeScript, Vite, Tailwind CSS, Zustand (for state management), Axios.
*   **Backend:** FastAPI, Python 3.11+, SQLAlchemy, Alembic (for migrations).
*   **AI/ML:** A custom agent orchestrator. Tools leverage SentenceTransformers (e.g., `all-MiniLM-L6-v2`) and ChromaDB for RAG.
*   **Infrastructure:** Docker, Redis, SQLite (dev), PostgreSQL (prod-ready).

## 3. System Architecture

The application follows a decoupled frontend/backend architecture. The frontend is a Single Page Application (SPA) built with React, and the backend is a FastAPI service. During development, the Vite dev server proxies API and WebSocket requests to the backend to avoid CORS issues.

### 3.1. Backend Architecture

The backend is a highly modular, production-grade FastAPI application.

*   **Core Component (`CustomerSupportAgent`)**: The heart of the backend is the `CustomerSupportAgent` (`backend/app/agents/chat_agent.py`). It is a sophisticated orchestrator that manages the entire conversation flow, tool usage, and state.

*   **Distributed Session Management**: A key architectural pillar is the externalized session management system, designed for horizontal scalability.
    *   **Pluggable Backends**: It uses a `SessionStore` abstraction (`backend/app/session/session_store.py`) with two implementations: a simple `InMemorySessionStore` for development and a production-grade `RedisSessionStore`.
    *   **High-Performance & Consistency**: The Redis store uses atomic Lua scripts for operations, features an in-memory L1 cache (`TTLCache`) to reduce latency, and employs Redis-based distributed locks (`backend/app/session/distributed_lock.py`) to prevent race conditions in a multi-instance environment.
    *   **Security**: Session data can be encrypted at rest, configured via `backend/app/config.py`.

*   **Extensible Tooling Architecture**: The system features a dynamic and resilient tool-use framework.
    *   **Tool Registry**: Instead of hardcoding tools, the agent uses a `ToolRegistry` (`backend/app/tools/registry.py`) to dynamically load and initialize tools based on configuration. This makes adding new capabilities (like the implemented CRM, Billing, and Inventory tools) straightforward.
    *   **Standardized Contracts**: All tools inherit from a `BaseTool` (`backend/app/tools/base_tool.py`) and return a standardized `ToolResult`, ensuring predictable integrations.
    *   **Resilience**: Tool calls are wrapped with resilience patterns like retries and circuit breakers (via `backend/app/tools/tool_call_wrapper.py`), making the system robust against transient failures of external services.

*   **Configuration-Driven System**: The entire backend's behavior is controlled by a comprehensive, type-safe configuration system in `backend/app/config.py`. This Pydantic-based setup allows for easy management of different environments, feature flags, and fine-grained tuning of all components, from the database pool to session encryption keys.

*   **Robust Application Lifecycle**: The `app/main.py` entry point defines a clean application lifecycle with a `lifespan` manager. This ensures graceful startup (including database checks, agent initialization) and shutdown. It also registers a full-featured middleware pipeline for rate limiting, request timing, correlation IDs, and global error handling.

### 3.2. Frontend Architecture

The frontend is a modern React application built with Vite and TypeScript.

*   **Component-Based UI**: The UI is built with React components located in `frontend/src/components`. The main view is the `ChatInterface.tsx`.
*   **Hook-Based Logic**: Core business logic and state management for the chat are encapsulated in custom hooks, primarily `useChat.ts`. This hook orchestrates API calls and WebSocket events.
*   **Service Layer**: API and WebSocket communications are abstracted into singleton services (`frontend/src/services/api.ts` and `frontend/src/services/websocket.ts`), keeping network logic separate from UI components.
*   **State Management**: The primary chat state is managed within the `useChat` hook using `useState`.
*   **Styling**: Tailwind CSS is used for styling, with custom configurations in `tailwind.config.js`.

## 4. Key Workflows & Data Flows

### 4.1. Session Initialization and WebSocket Connection

This workflow is critical for starting a conversation and was the source of a key bug.

1.  **Frontend**: The `ChatInterface` component mounts and, via the `useChat` hook, calls `initializeSession()`.
2.  **Frontend**: `initializeSession` makes a `POST` request to the backend's `/api/sessions` endpoint.
3.  **Backend**: The `create_session` route in `sessions.py` creates a new session in the database and returns a `SessionResponse` object.
4.  **Data Contract (Compatibility Fix)**: The backend returns a JSON object with `snake_case` keys (e.g., `{"session_id": "sess_..."}`). The frontend was previously expecting `camelCase` keys (`sessionId`), causing it to read `undefined` for the session ID. The type definition in `frontend/src/types/index.ts` and the property access in `frontend/src/hooks/useChat.ts` have been **corrected** to use `session_id`.
5.  **Frontend**: After successfully receiving the `session_id`, the `useChat` hook immediately calls `websocket.connect(session_id)`.
6.  **Frontend**: The `websocket.ts` service constructs the connection URL: `ws://<host>/ws?session_id=<session_id>`.
7.  **Backend**: The `websocket_endpoint` in `websocket.py` receives the connection, validates the `session_id`, and begins the real-time communication loop.

### 4.2. Sending a Message

Messages can be sent via HTTP (as a fallback) or WebSocket.

*   **WebSocket (Primary)**:
    1.  The user submits a message.
    2.  `useChat.ts` calls `websocket.sendMessage(content, attachments)`.
    3.  The `websocket.ts` service sends a JSON payload of `type: "message"` over the active WebSocket connection.
    4.  The backend's `websocket_endpoint` receives the message and streams back the agent's response in chunks (`type: "text"`, `type: "sources"`, etc.), culminating in a `type: "complete"` message.

*   **HTTP (Fallback)**:
    1.  If the WebSocket is not connected, `useChat.ts` calls `api.sendMessage(sessionId, content, attachments)`.
    2.  `api.ts` sends a `POST` request with `FormData` to `/api/chat/sessions/{sessionId}/messages`.
    3.  The backend's `send_message` route processes the entire request via `agent.process_message` and returns a complete `ChatResponse` once finished.

## 5. Development Conventions

The project follows modern development conventions. The current codebase should be considered the source of truth for all patterns.

*   **Code Style:**
    *   **Python:** PEP 8, extensive type hints, and a focus on readable, modular code.
    *   **TypeScript:** The project is set up with ESLint and Prettier for code quality.
*   **Testing:**
    *   **Backend**: The testing framework is in place using `pytest`. Run tests with `./scripts/run_tests.sh`.
    *   **Frontend**: The testing framework uses `vitest`. Run tests with `npm test`.
*   **Architecture Patterns:**
    *   **Service-Oriented**: The codebase demonstrates a clear separation of concerns between API routes (`app/api`), agent logic (`app/agents`), tools (`app/tools`), and services (`app/services`). This pattern should be followed for new features.
    *   **Adding New Tools**: To add a new tool, create a new class inheriting from `BaseTool`, implement its logic, and register it in the `ToolRegistry` (`app/tools/registry.py`). Enable it via configuration in `app/config/tool_settings.py`.
    *   **Configuration**: All new features should be configurable through `app/config.py` and `app/config/tool_settings.py`. Avoid hardcoded values.
    *   **Session State**: All session-related state must be stored and retrieved through the `SessionStore` abstraction to ensure compatibility with distributed deployments. Do not store state directly in the agent instance.

## 6. Building and Running

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
