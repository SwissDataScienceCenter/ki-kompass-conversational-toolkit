import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse

from conversational_toolkit import __version__
from conversational_toolkit.api.auth.base import AuthProvider
from conversational_toolkit.api.auth.session_cookie_provider import SessionCookieProvider
from conversational_toolkit.api.routes.api import create_api_router
from conversational_toolkit.conversation_database.controller import (
    ConversationalToolkitController,
)

DIST_FOLDER = str(Path(__file__).parents[3] / "dist")


def create_app(
    controller: ConversationalToolkitController,
    auth_provider: AuthProvider | None = None,
    allow_origins: list[str] | None = None,
    dist_path: str = DIST_FOLDER,
    env: str = os.getenv("ENV", "local"),
) -> FastAPI:
    if auth_provider is None:
        auth_provider = SessionCookieProvider(
            controller=controller,
            secret_key=os.getenv("SECRET_KEY", "1234567890"),
            algorithm="HS256",
            env=env,
        )

    if allow_origins is None:
        allow_origins = ["http://localhost:3000", "http://localhost:8080"]

    app = FastAPI(docs_url=None)

    app.add_middleware(
        CORSMiddleware,  # type: ignore
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    auth_provider.bind_to_app(app)

    api_router = create_api_router(controller, auth_provider)
    app.include_router(api_router)

    @app.get("/", response_model=None)
    @app.get("/c/{path:path}", response_model=None)
    async def root() -> FileResponse | JSONResponse:
        if os.path.isfile(os.path.join(dist_path, "index.html")):
            return FileResponse(os.path.join(dist_path, "index.html"))
        else:
            return JSONResponse(content={"version": __version__})

    if os.path.exists(dist_path):
        app.mount(
            "/",
            StaticFiles(directory=dist_path),
            name="dist",
        )

    return app
