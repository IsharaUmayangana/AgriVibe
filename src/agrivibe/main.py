from fastapi import FastAPI

from agrivibe.api import knowledge
from agrivibe.api import ask

app = FastAPI(
    title="AgriVibe API",
    description="AgriVibe is a multi-agent agriculture knowledge system for Sri Lanka",
    version="0.1.0",
)

app.include_router(knowledge.router)
app.include_router(ask.router)

@app.get("/health", tags=["Health", "System"])
def health() -> dict[str, str]:
    return {"status": "ok"}

