"""Domain layer — the innermost ring.

Pure business concepts. MUST NOT import FastAPI, SQLAlchemy, Redis, LangGraph,
or any provider SDK. If you find yourself needing one here, you actually need a
new port.
"""
