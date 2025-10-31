# No database migration needed for Phase 1
# Just restart the application

uvicorn backend.app.main:app --reload
