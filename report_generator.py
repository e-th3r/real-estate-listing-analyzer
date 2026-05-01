from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from env_loader import load_project_env
from langchain_chroma import Chroma

load_project_env()
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from vector_store import search_alternatives_with_scores


REPORT_SYSTEM_PROMPT = (
    "Ты — опытный аналитик рынка недвижимости. На основании описания "
    "проверяемой квартиры и набора похожих объявлений ты помогаешь покупателю "
    "понять, насколько выгодна сделка, и в чём преимущества и недостатки "
    "варианта по сравнению с конкурентами. Отвечай по-русски, конкретно, "
    "без воды. Все числовые сравнения приводи в рублях и ₽/м²."
)

REPORT_HUMAN_PROMPT = (
    "## Проверяемая квартира\n"
    "{target}\n\n"
    "## Похожие объявления (топ-{k} по семантическому поиску)\n"
    "{alternatives}\n\n"
    "## Что нужно сделать\n"
    "Составь краткий отчёт строго по разделам:\n"
    "1. **Вывод** (одна фраза: выгодная / средняя / переоценённая сделка).\n"
    "2. **Цена и ₽/м²** — сравни с медианой и диапазоном по похожим.\n"
    "3. **Плюсы на фоне конкурентов** — пунктами.\n"
    "4. **Минусы и риски** — пунктами, ссылайся на конкретные конкурирующие объявления.\n"
    "5. **На что обратить внимание перед сделкой** — 2–3 пункта.\n"
)

REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REPORT_SYSTEM_PROMPT),
        ("human", REPORT_HUMAN_PROMPT),
    ]
)


@dataclass
class TargetListing:
    """Описание квартиры, которую проверяет пользователь.

    Если нет полной структурированной информации (например, на вход подан
    только текст), достаточно `description`.
    """

    description: str
    listing_id: Optional[str] = None
    url: Optional[str] = None
    price_rub: Optional[float] = None
    total_area_m2: Optional[float] = None
    floor: Optional[int] = None
    floors_total: Optional[int] = None
    images: List[str] = field(default_factory=list)

    @property
    def price_per_m2(self) -> Optional[float]:
        if self.price_rub and self.total_area_m2:
            return float(self.price_rub) / float(self.total_area_m2)
        return None


def _format_money(value: Optional[float]) -> str:
    if value is None:
        return "н/д"
    return f"{value:,.0f} ₽".replace(",", " ")


def _format_listing_brief(doc: Document, score: Optional[float] = None) -> str:
    md = doc.metadata
    price = md.get("price_rub")
    area = md.get("total_area_m2")
    ppm = md.get("price_per_m2")
    floor = md.get("floor")
    floors_total = md.get("floors_total")
    score_part = f" | similarity_distance={score:.3f}" if score is not None else ""
    snippet = doc.page_content.strip()
    if len(snippet) > 600:
        snippet = snippet[:600].rstrip() + "..."
    return (
        f"- URL: {md.get('url')}{score_part}\n"
        f"  Цена: {_format_money(price)} | Площадь: {area} м² | "
        f"{_format_money(ppm)}/м²\n"
        f"  Этаж: {floor}/{floors_total}\n"
        f"  Описание: {snippet}"
    )


def _format_target_block(target: TargetListing) -> str:
    lines: List[str] = []
    if target.url:
        lines.append(f"URL: {target.url}")
    if target.price_rub is not None:
        lines.append(f"Цена: {_format_money(target.price_rub)}")
    if target.total_area_m2 is not None:
        lines.append(f"Площадь: {target.total_area_m2} м²")
    if target.price_per_m2 is not None:
        lines.append(f"Цена за м²: {_format_money(target.price_per_m2)}/м²")
    if target.floor is not None and target.floors_total is not None:
        lines.append(f"Этаж: {target.floor}/{target.floors_total}")
    snippet = target.description.strip()
    if len(snippet) > 1500:
        snippet = snippet[:1500].rstrip() + "..."
    lines.append(f"Описание:\n{snippet}")
    return "\n".join(lines)


TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"


def build_default_llm(
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Создаёт ChatOpenAI поверх любого OpenAI-совместимого провайдера.

    Приоритет источников конфигурации:

    1. Явные аргументы функции.
    2. Переменные окружения `LISTING_LLM_*` (модель / base_url / api_key).
    3. Together AI: если задан `TOGETHER_API_KEY` — base_url и модель
       выставляются автоматически (`https://api.together.xyz/v1`,
       serverless-модель DeepSeek-V3 по умолчанию).
    4. Fallback: `OPENAI_API_KEY` + дефолтная модель `gpt-4o-mini`.

    Локальные серверы (LM Studio / Ollama) задаются через
    `LISTING_LLM_BASE_URL=http://localhost:1234/v1`.
    """
    together_key = os.environ.get("TOGETHER_API_KEY")

    resolved_base_url = (
        base_url
        or os.environ.get("LISTING_LLM_BASE_URL")
        or (TOGETHER_BASE_URL if together_key else None)
    )
    resolved_api_key = (
        api_key
        or os.environ.get("LISTING_LLM_API_KEY")
        or together_key
        or os.environ.get("OPENAI_API_KEY")
    )

    is_together = resolved_base_url == TOGETHER_BASE_URL
    default_model = TOGETHER_DEFAULT_MODEL if is_together else "gpt-4o-mini"
    resolved_model = model or os.environ.get("LISTING_LLM_MODEL") or default_model

    kwargs: Dict[str, Any] = {"model": resolved_model, "temperature": temperature}
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    elif resolved_base_url:
        kwargs["api_key"] = "local"
    return ChatOpenAI(**kwargs)


def build_report_prompt(
    target: TargetListing,
    alternatives: Sequence[tuple[Document, float]],
) -> List[Any]:
    formatted_alts = "\n\n".join(
        _format_listing_brief(doc, score) for doc, score in alternatives
    )
    return REPORT_PROMPT.format_messages(
        target=_format_target_block(target),
        alternatives=formatted_alts or "(не найдено похожих объявлений)",
        k=len(alternatives),
    )


def generate_report(
    target: TargetListing,
    vector_store: Chroma,
    llm: BaseChatModel,
    *,
    k: int = 5,
) -> str:
    exclude = [target.listing_id] if target.listing_id else None
    pairs = search_alternatives_with_scores(
        vector_store,
        query=target.description,
        k=k,
        exclude_ids=exclude,
    )
    messages = build_report_prompt(target, pairs)
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        # Некоторые модели возвращают список content blocks.
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)
