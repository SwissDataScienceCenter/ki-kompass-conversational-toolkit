from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse


from conversational_toolkit.api.auth.base import AuthProvider
from conversational_toolkit.conversation_database.controller import (
    ConversationalToolkitController,
    ClientMessage,
    MessageInput,
    ClientConversation,
    ConversationInput,
    ReactionInput,
)
from conversational_toolkit.conversation_database.data_models.conversation import Conversation
from conversational_toolkit.conversation_database.data_models.reaction import Reaction


def create_api_router(
    controller: ConversationalToolkitController,
    auth_provider: AuthProvider,
) -> APIRouter:
    api_router = APIRouter(prefix="/api/v1")

    async def check_owner(conversation_id: str, user_id: str) -> None:
        try:
            conversation = await controller.conversation_db.get_conversation_by_id(conversation_id)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
        if conversation.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="User doesn't have access to this conversation"
            )

    @api_router.post("/messages")
    async def post_user_message(
        user_input: MessageInput, user_id: str = Depends(auth_provider.get_current_user_id)
    ) -> ClientMessage:
        if user_input.conversation_id:
            await check_owner(user_input.conversation_id, user_id)
        return await controller.process_new_message(user_id=user_id, user_input=user_input)

    @api_router.post("/messages/stream")
    async def post_user_message_stream(
        user_input: MessageInput, user_id: str = Depends(auth_provider.get_current_user_id)
    ) -> StreamingResponse:
        if user_input.conversation_id:
            await check_owner(user_input.conversation_id, user_id)
        return StreamingResponse(
            controller.process_new_message_stream(user_id=user_id, user_input=user_input),  # type: ignore
            media_type="json/event-stream",
        )

    @api_router.get("/conversations")
    async def get_conversations_metadata(
        user_id: str = Depends(auth_provider.get_current_user_id),
    ) -> list[Conversation]:
        return await controller.get_conversations_data_by_user_id(user_id)

    @api_router.get("/conversations/{conversation_id}")
    async def get_conversation(
        conversation_id: str, user_id: str = Depends(auth_provider.get_current_user_id)
    ) -> ClientConversation:
        await check_owner(conversation_id, user_id)
        return await controller.get_conversation_by_id(conversation_id)

    @api_router.put("/conversations/{conversation_id}")
    async def update_conversation(
        conversation_id: str,
        conversation_updates: ConversationInput,
        user_id: str = Depends(auth_provider.get_current_user_id),
    ) -> Conversation:
        await check_owner(conversation_id, user_id)
        return await controller.update_conversation(conversation_id, conversation_updates)

    @api_router.delete("/conversations/{conversation_id}")
    async def delete_conversation(
        conversation_id: str, user_id: str = Depends(auth_provider.get_current_user_id)
    ) -> bool:
        await check_owner(conversation_id, user_id)
        return await controller.delete_conversation(conversation_id)

    @api_router.get("/conversations/{conversation_id}/messages")
    async def get_messages(
        conversation_id: str, user_id: str = Depends(auth_provider.get_current_user_id)
    ) -> list[ClientMessage]:
        await check_owner(conversation_id, user_id)
        return await controller.get_messages_by_conversation_id(conversation_id)

    @api_router.post("/conversations/{conversation_id}/messages/{message_id}/reactions")
    async def post_reaction(
        reaction_input: ReactionInput,
        conversation_id: str,
        message_id: str,
        user_id: str = Depends(auth_provider.get_current_user_id),
    ) -> Reaction:
        await check_owner(conversation_id, user_id)
        return await controller.add_reaction(reaction_input=reaction_input, message_id=message_id, user_id=user_id)

    return api_router
