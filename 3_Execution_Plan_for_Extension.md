### Overview

This execution plan divides the work into logically independent, testable phases so an AI coding agent (or human engineer) can implement each change with minimal supervision and minimal rework. Each phase produces self-contained artifacts, tests, and migration/rollback steps. Phases are ordered to minimize downstream rework and to make each deliverable usable incrementally. Deliverables covered: (1) async BaseTool + ToolResult refactor, (2) ToolRegistry + agent initialization refactor, (3) RedisSessionStore adapter and session externalization, (4) example CRMLookupTool (template) + tests, (5) MessageAudit model + write-after-process logic. Each phase lists files to add/update, explicit change descriptions, and a checklist to validate completion.

Keep these priorities while implementing:
- Safety first: add tests before changing runtime behavior.
- Backwards-compatible toggles: add feature flags with default OFF before switching live behavior.
- Incremental rollout: implement adapters so old sync semantics can be supported during transition.
- Observability: instrument new code with structured logs and telemetry spans (request_id, session_id, tool_name) from the start.

---

### Phase 0 — Prep, tests and safety scaffolding (prework, immutable baseline)

Goal: Create test harnesses and scaffolding so refactors are verifiable and reversible.

Files to add/update
- Add: tests/conftest.py  
  - Provide pytest fixtures: test_db_engine (sqlite in-memory), test_sessionmaker, fake_cache (in-memory dict), settings_override.
- Add: tests/test_tool_contract.py  
  - Tests that assert BaseTool abstract methods presence and that ToolResult serialization works.
- Add: tests/test_agent_process_message_smoke.py  
  - Minimal smoke test invoking CustomerSupportAgent.process_message with mocked tools (mocks that implement old sync contract).
- Update: backend/app/tools/base_tool.py (minor docstring only)  
  - Add deprecation note and link to new async contract (no behavior change yet).
- Add: scripts/run_tests.sh  
  - Run pytest with environment that uses test fixtures and fails on any warnings.

Change description
- Provide automated tests to run before and after each phase. These tests must pass on current main to establish a baseline.
- Keep old code untouched for now; tests should run against current code and assert existing behaviors.

Checklist
- [ ] tests/conftest.py added and fixtures run without error.
- [ ] test_tool_contract.py passes on current main (should pass if tests verify presence of current base_tool).
- [ ] test_agent_process_message_smoke.py passes with mocked dependencies.
- [ ] run_tests.sh executes and returns exit 0 on baseline.

Rollback
- No code changes to production paths; safe to revert by deleting added test files.

---

### Phase 1 — Standardize tool contract: ToolResult + async BaseTool (Deliverable 1)

Goal: Introduce a standardized async tool contract and ToolResult return type, preserve backwards compatibility with adapter wrappers, and add unit tests.

Files to add/update
- Update: backend/app/tools/base_tool.py  
  - Add ToolResult dataclass/pydantic model with fields: success: bool, data: dict, metadata: dict, error: Optional[str].  
  - Replace or extend BaseTool to declare abstract async methods: async initialize(self), async cleanup(self). Add guidance for implementing async public methods. Keep legacy sync methods supported via adapters and mark deprecated.
- Add: backend/app/tools/tool_adapters.py  
  - Provide helper functions: sync_to_async_adapter(sync_fn) and ensure_tool_async(tool_instance) that wrap old sync methods into async-compatible coroutines using run_in_executor when necessary.
- Update: backend/app/tools/rag_tool.py, memory_tool.py, escalation_tool.py  
  - Implement minor changes to return ToolResult where public methods are used by agent. Introduce async wrapper methods (e.g., async add_documents_async that wraps existing add_documents). Do not remove old APIs yet.
- Add: tests/test_tool_async_contract.py  
  - Unit tests for ToolResult creation, serialization, and that adapters correctly wrap sync tool methods into async coroutines.

Change description
- Introduce ToolResult as canonical tool output and make BaseTool an async-first interface.
- Provide adapters to maintain compatibility so existing callers can be updated incrementally.
- Update concrete tools to expose async variants (without removing synchronous APIs) and to return ToolResult from async methods used by agent.

Checklist
- [ ] ToolResult dataclass added and importable from tools package.
- [ ] BaseTool declares async initialize/cleanup in abstract base and documents contract.
- [ ] tool_adapters.py implemented and unit-tested.
- [ ] rag_tool, memory_tool, escalation_tool expose async methods and return ToolResult via async wrappers.
- [ ] tests/test_tool_async_contract.py passes.

Rollback
- Revert to prior base_tool.py and remove adapters; old APIs remain unmodified because adapter and new code are additive.

Notes for the agent/caller
- Do not change CustomerSupportAgent to call async tool methods until Phase 2. Phase 1 only introduces the types and wrappers.

---

### Phase 2 — ToolRegistry and configuration-driven registration (Deliverable 2)

Goal: Replace static tool instantiation with a registry and runtime factory pattern, controlled by settings flags. Keep old initialization available behind a toggle for easy rollback.

Files to add/update
- Add: backend/app/tools/registry.py  
  - Implement ToolRegistry class with: register(name, factory_fn), create_tools(settings, dependencies) that instantiates enabled tools using settings flags and dependency injection (db, cache, http client).
- Update: backend/app/agents/chat_agent.py  
  - Replace hard-coded tool creation in _initialize with a call to ToolRegistry.create_tools. Support a setting AGENT_TOOL_REGISTRY_MODE = "legacy"|"registry" to switch behavior; default "legacy" until rollout completes.
- Add: backend/app/config/tool_settings.py  
  - Provide Pydantic settings entries for tool toggles (ENABLE_RAG, ENABLE_MEMORY, ENABLE_ESCALATION, ENABLE_CRM, ENABLE_BILLING) and per-tool configs (ENDPOINTS, TIMEOUTS).
- Add: tests/test_registry.py  
  - Unit tests that registry instantiates only enabled tools and that factory receives dependencies.

Change description
- ToolRegistry centralizes tool factories and isolates dependencies, enabling easier plugin addition and conditional enabling.
- Keeping AGENT_TOOL_REGISTRY_MODE ensures the system can run in legacy mode until all agent calls are updated.

Checklist
- [ ] registry.py added with docstrings and typed factory signatures.
- [ ] chat_agent._initialize refactored to use registry when AGENT_TOOL_REGISTRY_MODE == "registry".
- [ ] tool_settings.py added and integrated into application settings.
- [ ] tests/test_registry.py passes.
- [ ] Integration smoke runs: start app in registry mode and validate health endpoint.

Rollback
- Set AGENT_TOOL_REGISTRY_MODE back to "legacy"; registry code is additive and safe to remove if needed.

---

### Phase 3 — Standardize tool usage in agent (async update) and telemetry wrapper

Goal: Change CustomerSupportAgent to call async tool methods uniformly, instrument calls with telemetry/logging, and adopt ToolResult semantics. This is the critical behavioral update; perform after Phase 1 and Phase 2 tests pass.

Files to add/update
- Update: backend/app/agents/chat_agent.py  
  - Replace synchronous tool calls with awaited async methods returning ToolResult. Use tool_call_wrapper (see next file) to handle retries, circuit breakers, logging, and telemetry spans.
  - Update process_message to collect ToolResult.metadata into AgentResponse.tool_metadata.
  - Add deprecation warnings for legacy behavior if synchronous tool methods were still in use.
- Add: backend/app/tools/tool_call_wrapper.py  
  - Implement decorator/context manager tool_call_wrapper(tool_name, request_id, session_id, retry_policy, circuit_breaker_config) that:
    - Starts OpenTelemetry span named tool.{tool_name}
    - Emits structured logs before and after call including request_id, session_id, tool_name
    - Applies exponential backoff retries for transient errors (configurable) and a circuit-breaker per tool
    - Converts exceptions into ToolResult(success=False, error=...)
- Update: backend/app/main.py (minimal)  
  - Ensure request_id middleware sets request_id in request.state and propagate to agent calls.
- Add: tests/test_agent_tool_integration.py  
  - Integration tests with test registry and mocked tools returning ToolResults (success and failure), asserting agent behavior, telemetry wrapper invocation, and that tool_metadata is stored in response.

Change description
- Agent now consistently awaits tools and consumes ToolResult. The wrapper provides standardized resiliency and observability.
- Keep legacy sync adapter available as fallback but log deprecation warnings.

Checklist
- [ ] chat_agent updated to await tool async methods and to collect ToolResult metadata.
- [ ] tool_call_wrapper implemented and used on all tool calls.
- [ ] tests/test_agent_tool_integration.py passes.
- [ ] Service startup in registry mode and processing a message returns expected AgentResponse.tool_metadata shapes.

Rollback
- Revert AGENT_TOOL_REGISTRY_MODE to "legacy" and/or configure feature flag to use legacy synchronous calls if critical issues occur.

---

### Phase 4 — SessionStore adapter and RedisSessionStore (Deliverable 3)

Goal: Externalize ephemeral AgentContext into a pluggable SessionStore so contexts survive across instances and enable consistent counters/flags. Provide both InMemorySessionStore and RedisSessionStore implementations.

Files to add/update
- Add: backend/app/session/session_store.py  
  - Define SessionStore interface with methods: get(session_id), set(session_id, context_dict), update(session_id, patch_dict, atomic=False), delete(session_id), list_active().
- Add: backend/app/session/in_memory_session_store.py  
  - Provide simple dict-based implementation for local dev and tests.
- Add: backend/app/session/redis_session_store.py  
  - Implement Redis-backed store using Redis HASH or JSON (if RedisJSON available); include atomic increment methods for message_count using HINCRBY or Lua script for complex updates.
- Update: backend/app/agents/chat_agent.py  
  - Replace local in-process self.contexts with calls to SessionStore. Provide in-memory caching just for hot path if desired but persist authoritative context to SessionStore after updates.
- Add: tests/test_session_store.py  
  - Unit tests for InMemorySessionStore and RedisSessionStore behavior; ensure atomic increments work as expected under concurrent access simulation.

Change description
- SessionStore abstracts the storage; the agent uses it to persist counters (message_count), flags (escalated), and per-session ephemeral cache keys.
- RedisSessionStore must use atomic operations for increments and updates to avoid race conditions when multiple agent instances process messages for the same session.

Checklist
- [ ] session_store.py interface added and documented.
- [ ] in_memory_session_store.py implemented and used by default in dev.
- [ ] redis_session_store.py implemented; unit tested with a local test Redis instance.
- [ ] chat_agent updated to use SessionStore get/set/update operations.
- [ ] tests/test_session_store.py passes for both stores.

Rollback
- Revert config to use InMemorySessionStore by setting USE_SHARED_CONTEXT = false.

Migration notes
- Add setting USE_SHARED_CONTEXT (default false). Rollout: enable in staging first, then production once validated.

---

### Phase 5 — CRMLookupTool template + registration + tests (Deliverable 4)

Goal: Provide a complete example of adding a domain tool that adheres to the new async tool contract, includes retries, circuit breaker, telemetry, registration, and unit/integration tests.

Files to add/update
- Add: backend/app/tools/crm_tool.py  
  - Implements BaseTool with async initialize/cleanup and public async method lookup_customer(customer_id) returning ToolResult containing customer profile, account status, and metadata (source, fetched_at). Accepts injected HTTP client, supports auth, retry and circuit breaker.
- Update: backend/app/tools/registry.py  
  - Add factory registration for 'crm' tool, controlled by ENABLE_CRM_TOOL setting.
- Add: tests/test_crm_tool.py  
  - Unit tests mocking HTTP responses: success, 404, 500 (assert retry/circuit breaker behavior and ToolResult fields).
- Add: tests/integration/test_agent_with_crm.py  
  - Agent integration test that registers CRMLookupTool (registry mode) and asserts that process_message calls CRMLookupTool and that result is present in AgentResponse.tool_metadata and persisted to MessageAudit (phase 6).

Change description
- CRMLookupTool demonstrates required structure for external connectors: per-call telemetry, retries via decorator, structured ToolResult, credential handling from settings, and a factory function for registry.
- Tests ensure behavior in success and failure modes, mirroring production scenarios.

Checklist
- [ ] crm_tool.py added and adheres to async BaseTool signature.
- [ ] registry updated so ENABLE_CRM_TOOL toggles registration.
- [ ] tests/test_crm_tool.py passes locally.
- [ ] integration test confirms agent receives CRM data and stores tool metadata.

Rollback
- Disable ENABLE_CRM_TOOL in settings to remove runtime usage.

---

### Phase 6 — MessageAudit model and persistence (Deliverable 5)

Goal: Add a MessageAudit SQLAlchemy model and persist an audit row per processed message capturing tool calls, selected RAG sources, escalation state, and PII flags.

Files to add/update
- Add: backend/app/models/message_audit.py  
  - SQLAlchemy model: id (uuid), session_id, request_id, timestamp, message_text (redacted/pointer), tools_called (JSON), selected_sources (JSON), escalation_ticket_id, pii_redacted boolean, raw_payload_s3_key (optional).
- Update: backend/app/models/__init__.py  
  - Export new model and include in alembic autogenerate scope.
- Update: backend/app/agents/chat_agent.py  
  - After building AgentResponse and before sending/streaming, create and persist MessageAudit row using DB session. If message payload larger than X bytes, store full payload in S3 and save s3 key.
- Add: backend/alembic/versions/xxxx_add_message_audit_table.py  
  - Alembic migration to add the message_audit table.
- Add: tests/test_message_audit.py  
  - Unit test to ensure audit rows are created with proper fields and that redaction flag is set when PII detected (mock PII detection to return True/False).

Change description
- Audit model stores structured artifacts for compliance. agent writes audit rows synchronously within transaction scope after response generation to guarantee consistency.
- Use an async task/queue (optional) if audit writes cause latency; default synchronous write is acceptable at small scale and provides stronger guarantees.

Checklist
- [ ] message_audit model added and included in models exports.
- [ ] alembic migration created and applied in test DB.
- [ ] agent persists audit row per message, including tools_called and selected_sources.
- [ ] tests/test_message_audit.py passes.

Rollback
- If audit writes cause performance regressions, toggle AUDIT_WRITE_MODE = "sync"|"async" and set to "async" to queue writes to background worker. Provide a rollback path by switching to "async" or disabling audit temporarily.

---

### Validation and acceptance gates (apply after each phase)

For each phase, perform the following acceptance checks before proceeding:
1. Unit tests: all new and existing unit tests pass locally and in CI.
2. Integration smoke: start the app in a disposable environment (docker-compose or test VM) and exercise core endpoints. Validate health endpoints and one end-to-end message flow.
3. Performance regression: run a small load test (e.g., 50 concurrent sessions) to watch latency and error rate; confirm no regressions above pre-established thresholds.
4. Observability check: verify OpenTelemetry spans, and structured logs for the changed code path contain request_id and session_id.
5. Rollback dry run: confirm that toggling the relevant feature flag or reverting config returns behavior to previous baseline without code revert.

---

### Rollout strategy and timeline (high-level)

- Day 0–2: Phase 0 (tests & baseline). Gate: CI green.
- Day 3–6: Phase 1 (ToolResult + async BaseTool + adapters). Gate: unit tests and tool contract tests pass.
- Day 7–10: Phase 2 (Registry) + small integration. Gate: agent in registry mode starts; registry tests pass.
- Day 11–16: Phase 3 (Agent async usage + tool_call_wrapper). Gate: full integration tests, telemetry present, no functional regressions.
- Day 17–22: Phase 4 (SessionStore + Redis implementation). Gate: multi-instance test demonstrates consistent counters/flags across two instances.
- Day 23–28: Phase 5 (CRMLookupTool) + integration tests. Gate: CRM integration tests pass; agent produces tool metadata and audit entries.
- Day 29–34: Phase 6 (MessageAudit schema + persistence + migration). Gate: Alembic migration ran, audit rows produced in staging.
- Day 35–40: Hardening, load tests, and staggered production rollout behind feature flags.

Adjust the timeline according to team size and CI cadence. Each phase is independent enough to be implemented and rolled back using toggles.

---

### Final checklist before making production changes

- [ ] All phases’ unit and integration tests pass in CI with coverage checks.
- [ ] Feature flags present for all behavior-affecting changes (registry mode, USE_SHARED_CONTEXT, ENABLE_CRM_TOOL, AUDIT_WRITE_MODE). Default conservative values for production.
- [ ] Alembic migrations staged and tested in staging DB restore test.
- [ ] Redis instance provisioned and tested for SessionStore atomic operations.
- [ ] Observability configured (OTLP endpoint, Prometheus metrics, structured logs). Validate test spans and logs include request_id and session_id.
- [ ] Secret management configured for new credentials (CRM, Redis) and not stored in .env files.
- [ ] Runbook updated with rollback steps for each phase: feature-flag toggles, service restart, migration revert steps.

---

https://copilot.microsoft.com/shares/G7mgprttsZU1KQfHQfFfB 
