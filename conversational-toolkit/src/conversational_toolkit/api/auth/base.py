from abc import ABC, abstractmethod
from fastapi import Request, FastAPI


class AuthProvider(ABC):
    """Identifies the user of each request."""

    @abstractmethod
    def get_current_user_id(self, request: Request) -> str:
        """FastAPI dependency returning the user ID; raises a 401 'HTTPException' if unauthenticated."""

    @abstractmethod
    def bind_to_app(self, app: FastAPI) -> None:
        """Register the provider's routes and middleware; the frontend expects '/auth/refresh'."""
