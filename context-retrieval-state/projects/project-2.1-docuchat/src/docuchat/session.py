from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class FileChatSessionStore:
    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        safe_name = session_id.replace("/", "_")
        return self.root / f"{safe_name}.json"

    def load_messages(self, session_id: str, limit: int | None = None) -> list[BaseMessage]:
        path = self._path_for(session_id)
        if not path.exists():
            return []

        raw_messages = json.loads(path.read_text(encoding="utf-8"))
        messages = [self._deserialize(item) for item in raw_messages]

        if limit is None:
            return messages

        return messages[-limit:]

    def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        messages = self.load_messages(session_id)
        messages.extend([HumanMessage(content=user_message), AIMessage(content=assistant_message)])

        serialized = [self._serialize(message) for message in messages]
        self._path_for(session_id).write_text(
            json.dumps(serialized, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _serialize(message: BaseMessage) -> dict[str, str]:
        role = "assistant" if isinstance(message, AIMessage) else "human"
        return {"role": role, "content": str(message.content)}

    @staticmethod
    def _deserialize(payload: dict[str, str]) -> BaseMessage:
        if payload["role"] == "assistant":
            return AIMessage(content=payload["content"])
        return HumanMessage(content=payload["content"])
