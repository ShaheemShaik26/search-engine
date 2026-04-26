from fastapi import FastAPI

from searchengine.api.routes import router
from searchengine.core.config import settings

app = FastAPI(title=settings.app_name, version=settings.version)
app.include_router(router)
