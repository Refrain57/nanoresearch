"""User settings endpoints — model preferences."""

from __future__ import annotations

import uuid as _uuid

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from nanoresearch.server.middleware.auth import get_current_user
from nanoresearch.storage.repositories.user_settings_repo import UserSettingsRepository

router = APIRouter()


def _repo(request: Request) -> UserSettingsRepository:
    return UserSettingsRepository(request.app.state.session_factory)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ProviderIn(BaseModel):
    id: str | None = None
    name: str
    provider: str | None = None   # preset key: deepseek/openai/anthropic/dashscope/azure_openai/siliconflow/openai_compatible
    api_key: str | None = None    # None = keep existing; "" = clear
    api_base: str | None = None
    models: list[str] = []


class UserSettingsUpdate(BaseModel):
    # "" means "clear to null"; None means "don't touch"
    model: str | None = None
    fast_model: str | None = None
    max_iterations: int | None = None
    providers: list[ProviderIn] | None = None
    roles: dict[str, dict | None] | None = None
    ragas_generator_model: str | None = None
    ragas_evaluator_model: str | None = None
    ragas_embedding_model: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_key(key: str) -> str:
    if not key:
        return ""
    return ("*" * 8 + key[-4:]) if len(key) > 4 else "*" * len(key)


def _mask_providers(providers: list[dict]) -> list[dict]:
    return [
        {
            "id": p.get("id"),
            "name": p.get("name", ""),
            "provider": p.get("provider"),
            "api_base": p.get("api_base"),
            "api_key_set": bool(p.get("api_key")),
            "api_key_hint": _mask_key(p.get("api_key") or ""),
            "models": p.get("models", []),
        }
        for p in providers
    ]


def _merge_providers(existing: list[dict], incoming: list[ProviderIn]) -> list[dict]:
    """Merge provider list, preserving stored api_keys when client sends None."""
    existing_map = {p["id"]: p for p in existing if p.get("id")}
    result = []
    for p in incoming:
        pid = p.id or str(_uuid.uuid4())
        base = existing_map.get(pid, {})
        # api_key=None → keep existing; api_key="" → clear; api_key="sk-xxx" → update
        if p.api_key is None:
            api_key = base.get("api_key", "")
        else:
            api_key = p.api_key  # "" clears it, non-empty sets it
        result.append({
            "id": pid,
            "name": p.name,
            "provider": p.provider if p.provider is not None else base.get("provider"),
            "api_base": p.api_base,
            "models": p.models,
            "api_key": api_key,
        })
    return result


def _to_dict(row, defaults: dict) -> dict:
    extra = (row.extra if row else None) or {}
    return {
        "model": (row.model if row else None) or defaults.get("model"),
        "fast_model": extra.get("fast_model"),
        "max_iterations": (row.max_iterations if row else None) or defaults.get("max_iterations"),
        "providers": _mask_providers(extra.get("providers", [])),
        "roles": extra.get("roles") or {},
        "ragas_generator_model": extra.get("ragas_generator_model") or defaults.get("ragas_generator_model"),
        "ragas_evaluator_model": extra.get("ragas_evaluator_model") or defaults.get("ragas_evaluator_model"),
        "ragas_embedding_model": extra.get("ragas_embedding_model") or defaults.get("ragas_embedding_model"),
    }


def _defaults(request: Request) -> dict:
    cfg = getattr(request.app.state, "loop_config", None) or {}
    rag = getattr(request.app.state, "rag_settings", None)
    eval_cfg = getattr(rag, "eval", None) or {}
    return {
        "model": cfg.get("model"),
        "max_iterations": cfg.get("max_iterations"),
        "ragas_generator_model": getattr(eval_cfg, "generator_model", None) or "qwen-plus",
        "ragas_evaluator_model": getattr(eval_cfg, "evaluator_model", None) or "qwen-max",
        "ragas_embedding_model": getattr(getattr(rag, "embedding", None), "model", None) or "text-embedding-v3",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/settings/me")
async def get_settings(request: Request, uid: str = Depends(get_current_user)):
    row = await _repo(request).get(uid)
    return _to_dict(row, _defaults(request))


@router.put("/api/settings/me")
async def update_settings(
    request: Request,
    body: UserSettingsUpdate,
    uid: str = Depends(get_current_user),
):
    repo = _repo(request)
    row = await repo.get(uid)
    extra = dict(row.extra or {}) if row else {}

    update: dict = {}

    # Use __fields_set__ so None can mean "clear" (if explicitly sent) vs "not provided"
    sent = body.model_fields_set if hasattr(body, "model_fields_set") else body.__fields_set__

    if "model" in sent:
        update["model"] = body.model or None  # "" → None
    if "max_iterations" in sent:
        update["max_iterations"] = body.max_iterations

    extra_changed = False

    if "providers" in sent and body.providers is not None:
        extra["providers"] = _merge_providers(extra.get("providers", []), body.providers)
        extra_changed = True

    if "roles" in sent and body.roles is not None:
        extra["roles"] = body.roles
        extra_changed = True

    for field, key in [
        ("fast_model", "fast_model"),
        ("ragas_generator_model", "ragas_generator_model"),
        ("ragas_evaluator_model", "ragas_evaluator_model"),
        ("ragas_embedding_model", "ragas_embedding_model"),
    ]:
        if field in sent:
            val = getattr(body, field)
            extra[key] = val or None  # "" → None (clears), "model" → sets
            extra_changed = True

    if extra_changed:
        update["extra"] = extra

    if update:
        row = await repo.upsert(uid, **update)
        request.app.state.web_loops.pop(uid, None)

    return _to_dict(row, _defaults(request))


@router.get("/api/settings/available-models")
async def available_models(request: Request, uid: str = Depends(get_current_user)):
    models = getattr(request.app.state, "allowed_models", [])
    return {"models": models}
