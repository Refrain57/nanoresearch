"""Unified LLM configuration resolution for all model roles.

ModelFactory is a pure resolution layer — it reads three config sources
(config.json Config, settings.yaml Settings, UserSettings DB row) and
returns a ModelSpec for each role.  No LLM objects are instantiated here.

Usage
-----
spec = ModelFactory.resolve(
    ModelRole.INGESTION_LLM,
    config=app.state.config,          # from config.json
    rag_settings=app.state.rag_settings,  # from settings.yaml
    user_model=user_cfg.model,
    user_providers=(user_cfg.extra or {}).get("providers", []),
)
patched = ModelFactory.patch_settings(rag_settings, ModelRole.INGESTION_LLM, spec)
llm = LLMFactory.create(patched)
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nanoresearch.config.schema import Config, ProviderConfig
    from nanoresearch.rag.core.settings import Settings

logger = logging.getLogger(__name__)


class ModelRole(Enum):
    CHAT = "chat"
    INGESTION_LLM = "ingestion_llm"
    EMBEDDING = "embedding"
    VISION = "vision"
    EVAL_GENERATOR = "eval_generator"
    EVAL_EVALUATOR = "eval_evaluator"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved, ready-to-use model configuration."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    extra_headers: dict[str, str] | None = None
    provider: str | None = None


class ModelResolutionError(ValueError):
    """Raised when model resolution fails to produce an API key."""

    def __init__(
        self,
        message: str,
        *,
        sources_checked: list[str] | None = None,
        missing_role: str | None = None,
    ) -> None:
        self.sources_checked = sources_checked or []
        self.missing_role = missing_role
        super().__init__(message)


class ModelFactory:
    """Resolve LLM configuration for every call role from three config sources."""

    @classmethod
    def require_key(
        cls,
        role: ModelRole,
        *,
        config: "Config | None" = None,
        rag_settings: "Settings | None" = None,
        user_model: str | None = None,
        user_providers: list[dict] | None = None,
        **overrides: Any,
    ) -> ModelSpec:
        """Resolve a ModelSpec and raise ModelResolutionError if api_key is missing.

        The ``sources_checked`` field on the exception lists each source
        that was examined during resolution (in priority order).
        """
        spec = cls.resolve(
            role,
            config=config,
            rag_settings=rag_settings,
            user_model=user_model,
            user_providers=user_providers,
            **overrides,
        )
        if not spec.api_key:
            role_name = role.value
            sources: list[str] = []
            _providers = user_providers or []
            if _providers:
                sources.append("user_providers")
            if config:
                sources.append(f"config.json (role={role_name})")
            if rag_settings:
                sources.append(f"settings.yaml (role={role_name})")
            raise ModelResolutionError(
                f"No API key found for role '{role_name}' (model={spec.model!r})",
                sources_checked=sources or ["no sources available"],
            )
        return spec

    @classmethod
    def resolve(
        cls,
        role: ModelRole,
        *,
        config: "Config | None" = None,
        rag_settings: "Settings | None" = None,
        user_model: str | None = None,
        user_providers: list[dict] | None = None,
        mode: Literal["server", "local"] | None = None,
        **overrides: Any,
    ) -> ModelSpec:
        """Return a ModelSpec for the given role.

        Parameters
        ----------
        role:
            Which use-case this LLM serves.
        config:
            Loaded config.json Config object (may be None in tests).
        rag_settings:
            Loaded settings.yaml Settings object (may be None when RAG disabled).
        user_model:
            user_settings.model — user's preferred chat model.
        user_providers:
            user_settings.extra["providers"] list of dicts:
            [{name, api_key, api_base (opt), models: [] (opt)}]
        mode:
            Override deployment mode ("server" or "local"). None → auto-detect
            via get_mode().
        **overrides:
            - model_override (str): beats everything for model selection.
            - api_key / base_url: force credentials.
            - user_extra (dict): full user_settings.extra (for ragas_* fields).
        """
        from nanoresearch.config.loader import get_mode

        effective_mode = mode or get_mode()
        _providers = user_providers or []

        # server 模式：缺 user_providers 命中直接 raise，不读 config / rag_settings
        if effective_mode == "server":
            spec = cls._resolve_from_user_only(
                role=role,
                user_model=user_model,
                user_providers=_providers,
                **overrides,
            )
            if not spec.api_key:
                raise ModelResolutionError(
                    f"No API key for role '{role.value}' in server mode "
                    f"(user_settings.extra.providers empty or no match)",
                    sources_checked=["user_providers"],
                    missing_role=role.value,
                )
            return spec

        # local 模式 — 现有 dispatch 不变
        dispatch = {
            ModelRole.CHAT: cls._resolve_chat,
            ModelRole.INGESTION_LLM: cls._resolve_ingestion_llm,
            ModelRole.EMBEDDING: cls._resolve_embedding,
            ModelRole.VISION: cls._resolve_vision,
            ModelRole.EVAL_GENERATOR: cls._resolve_eval_generator,
            ModelRole.EVAL_EVALUATOR: cls._resolve_eval_evaluator,
        }
        logger.debug(
            "Resolving role=%s user_model=%s user_providers=%s has_config=%s has_settings=%s overrides=%s",
            role.value,
            user_model,
            len(_providers),
            config is not None,
            rag_settings is not None,
            sorted(overrides.keys()),
        )
        return dispatch[role](
            config=config,
            rag_settings=rag_settings,
            user_model=user_model,
            user_providers=_providers,
            **overrides,
        )

    # ------------------------------------------------------------------
    # Role resolvers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_chat(
        cls, *, config, rag_settings, user_model, user_providers, **overrides
    ) -> ModelSpec:
        """CHAT: model_override > user model > config.json defaults."""
        model = (
            overrides.get("model_override")
            or user_model
            or (config.agents.defaults.model if config else None)
            or "gpt-4o"
        )
        matched = cls._match_user_provider_by_model(model, user_providers)
        if matched:
            return ModelSpec(
                model=model,
                api_key=matched.get("api_key") or None,
                base_url=matched.get("api_base") or None,
                provider=matched.get("name") or "openai",
            )
        if config:
            api_key = config.get_api_key(model)
            base_url = config.get_api_base(model)
            provider_name = config.get_provider_name(model)
            p = config.get_provider(model)
            return ModelSpec(
                model=model,
                api_key=api_key or None,
                base_url=base_url or None,
                extra_headers=p.extra_headers if p else None,
                provider=provider_name,
            )
        return ModelSpec(model=model)

    @classmethod
    def _resolve_ingestion_llm(
        cls, *, config, rag_settings, user_model, user_providers, **overrides
    ) -> ModelSpec:
        """INGESTION_LLM: model from settings.yaml; key from user > config.json > settings.yaml."""
        model = (rag_settings.llm.model if rag_settings else None) or (
            config.agents.defaults.model if config else "gpt-4o"
        )
        rag_provider = rag_settings.llm.provider if rag_settings else None

        matched = cls._match_user_provider_by_model(model, user_providers)
        if matched:
            return ModelSpec(
                model=model,
                api_key=matched.get("api_key") or None,
                base_url=matched.get("api_base") or None,
                provider=matched.get("name") or "openai",
            )
        if config and rag_provider:
            cfg_p = cls._lookup_provider_config(config, rag_provider)
            if cfg_p and cfg_p.api_key:
                return ModelSpec(
                    model=model,
                    api_key=cfg_p.api_key,
                    base_url=cfg_p.api_base or (rag_settings.llm.base_url if rag_settings else None),
                    provider=rag_provider,
                )
        if rag_settings:
            return ModelSpec(
                model=rag_settings.llm.model,
                api_key=rag_settings.llm.api_key,
                base_url=rag_settings.llm.base_url,
                provider=rag_settings.llm.provider,
            )
        return ModelSpec(model=model)

    @classmethod
    def _resolve_embedding(
        cls, *, config, rag_settings, user_model, user_providers, **overrides
    ) -> ModelSpec:
        """EMBEDDING: model from settings.yaml; key from user > config.json > settings.yaml."""
        if not rag_settings:
            return ModelSpec(model="")
        emb = rag_settings.embedding
        matched = cls._match_user_provider_by_name(emb.provider, user_providers)
        if matched:
            return ModelSpec(
                model=emb.model,
                api_key=matched.get("api_key") or None,
                base_url=matched.get("api_base") or None,
                provider=emb.provider,
            )
        if config:
            cfg_p = cls._lookup_provider_config(config, emb.provider)
            if cfg_p and cfg_p.api_key:
                return ModelSpec(
                    model=emb.model,
                    api_key=cfg_p.api_key,
                    base_url=cfg_p.api_base or emb.base_url,
                    provider=emb.provider,
                )
        return ModelSpec(
            model=emb.model,
            api_key=emb.api_key,
            base_url=emb.base_url,
            provider=emb.provider,
        )

    @classmethod
    def _resolve_vision(
        cls, *, config, rag_settings, user_model, user_providers, **overrides
    ) -> ModelSpec:
        """VISION: fixed to settings.yaml vision_llm; key bridged from config.json."""
        if not rag_settings or not rag_settings.vision_llm or not rag_settings.vision_llm.enabled:
            return ModelSpec(model="")
        vl = rag_settings.vision_llm
        api_key = vl.api_key
        if not api_key and config:
            cfg_p = cls._lookup_provider_config(config, vl.provider)
            api_key = cfg_p.api_key if (cfg_p and cfg_p.api_key) else None
        return ModelSpec(
            model=vl.model,
            api_key=api_key,
            base_url=vl.base_url,
            provider=vl.provider,
        )

    @classmethod
    def _resolve_eval_generator(
        cls, *, config, rag_settings, user_model, user_providers, **overrides
    ) -> ModelSpec:
        """EVAL_GENERATOR: request > user_extra.ragas_generator_model > config > settings > default."""
        return cls._resolve_eval_role(
            role_extra_key="ragas_generator_model",
            default_model="qwen-plus",
            config=config,
            rag_settings=rag_settings,
            user_providers=user_providers,
            **overrides,
        )

    @classmethod
    def _resolve_eval_evaluator(
        cls, *, config, rag_settings, user_model, user_providers, **overrides
    ) -> ModelSpec:
        """EVAL_EVALUATOR: request > user_extra.ragas_evaluator_model > config > settings > default."""
        return cls._resolve_eval_role(
            role_extra_key="ragas_evaluator_model",
            default_model="qwen-max",
            config=config,
            rag_settings=rag_settings,
            user_providers=user_providers,
            **overrides,
        )

    @classmethod
    def _resolve_eval_role(
        cls,
        *,
        role_extra_key: str,
        default_model: str,
        config,
        rag_settings,
        user_providers,
        **overrides,
    ) -> ModelSpec:
        user_extra = overrides.get("user_extra") or {}
        model = (
            overrides.get("model_override")
            or user_extra.get(role_extra_key)
            or (config.agents.defaults.model if config else None)
            or (rag_settings.llm.model if rag_settings else None)
            or default_model
        )
        matched = cls._match_user_provider_by_model(model, user_providers)
        if matched:
            return ModelSpec(
                model=model,
                api_key=matched.get("api_key") or None,
                base_url=matched.get("api_base") or None,
                provider=matched.get("name") or "openai",
            )
        if config:
            api_key = config.get_api_key(model)
            base_url = config.get_api_base(model)
            if api_key:
                return ModelSpec(model=model, api_key=api_key, base_url=base_url, provider=config.get_provider_name(model))
        if rag_settings:
            return ModelSpec(
                model=model,
                api_key=rag_settings.llm.api_key,
                base_url=rag_settings.llm.base_url,
            )
        return ModelSpec(model=model)

    # ------------------------------------------------------------------
    # patch_settings: apply ModelSpec back into a Settings copy
    # ------------------------------------------------------------------

    @staticmethod
    def patch_settings(rag_settings: "Settings", role: ModelRole, spec: ModelSpec) -> "Settings":
        """Return a new Settings with spec's values applied to the correct sub-field.

        Allows LLMFactory.create(settings) and EmbeddingFactory.create(settings)
        to remain unchanged — they still receive a Settings object.
        """
        if role == ModelRole.INGESTION_LLM:
            new_llm = dataclasses.replace(
                rag_settings.llm,
                model=spec.model or rag_settings.llm.model,
                api_key=spec.api_key if spec.api_key is not None else rag_settings.llm.api_key,
                base_url=spec.base_url if spec.base_url is not None else rag_settings.llm.base_url,
            )
            return dataclasses.replace(rag_settings, llm=new_llm)

        if role == ModelRole.EMBEDDING:
            new_emb = dataclasses.replace(
                rag_settings.embedding,
                model=spec.model or rag_settings.embedding.model,
                api_key=spec.api_key if spec.api_key is not None else rag_settings.embedding.api_key,
                base_url=spec.base_url if spec.base_url is not None else rag_settings.embedding.base_url,
            )
            return dataclasses.replace(rag_settings, embedding=new_emb)

        if role == ModelRole.VISION and rag_settings.vision_llm:
            new_vl = dataclasses.replace(
                rag_settings.vision_llm,
                api_key=spec.api_key if spec.api_key is not None else rag_settings.vision_llm.api_key,
            )
            return dataclasses.replace(rag_settings, vision_llm=new_vl)

        return rag_settings

    @classmethod
    def _resolve_from_user_only(
        cls,
        *,
        role: ModelRole,
        user_model: str | None,
        user_providers: list[dict],
        **overrides: Any,
    ) -> ModelSpec:
        """Resolve ModelSpec from user_providers only (server mode).

        For CHAT / EVAL_*: model from override > user_model > first provider's first model.
        For INGESTION / EMBEDDING / VISION: caller must pass model_override OR
        we default to user_model; if still empty, leave spec.model="" (caller error).
        """
        model = (
            overrides.get("model_override")
            or user_model
            or ""
        )
        if not model and user_providers:
            for p in user_providers:
                ms = p.get("models") or []
                if ms:
                    model = ms[0]
                    break
        matched = cls._match_user_provider_by_model(model, user_providers) if model else None
        if not matched and user_providers:
            # fallback 第一个有 api_key 的
            matched = next((p for p in user_providers if p.get("api_key")), None)
        if matched:
            return ModelSpec(
                model=model or "",
                api_key=matched.get("api_key") or None,
                base_url=matched.get("api_base") or None,
                provider=matched.get("name") or None,
            )
        return ModelSpec(model=model or "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_user_provider_by_model(model: str, user_providers: list[dict]) -> dict | None:
        """Find user provider whose models[] contains model; fall back to first with api_key."""
        if not model or not user_providers:
            return None
        exact = next(
            (p for p in user_providers if model in (p.get("models") or [])),
            None,
        )
        if exact:
            return exact
        return next((p for p in user_providers if p.get("api_key")), None)

    @staticmethod
    def _match_user_provider_by_name(provider_name: str, user_providers: list[dict]) -> dict | None:
        """Find user provider by exact name match (case-insensitive)."""
        if not provider_name or not user_providers:
            return None
        lname = provider_name.lower()
        return next(
            (p for p in user_providers if (p.get("name") or "").lower() == lname),
            None,
        )

    @staticmethod
    def _lookup_provider_config(config: "Config", provider_name: str) -> "ProviderConfig | None":
        """Get ProviderConfig from config.json by provider name.

        Normalises the name (e.g. "openai-compat" → "openai_compat") and
        does a direct attribute lookup on ProvidersConfig.
        """
        if not provider_name or not config:
            return None
        normalised = provider_name.lower().replace("-", "_")
        p = getattr(config.providers, normalised, None)
        if p is not None and p.api_key:
            return p
        return None
