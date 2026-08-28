"""Targeted recovery for relevance decisions contaminated by a date prompt."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from yt_searchapi.prompts import (
    CLASSIFIER_PROMPT_VERSION,
    CompiledClassifierPrompt,
    TopicBrief,
    TopicExpansion,
    compile_classifier_prompt,
)
from yt_searchapi.state import CrawlProjectState, ProjectStateStore

_RESEARCH_DEFINITION_MARKER = "\nRESEARCH_DEFINITION_JSON\n"
_PUBLICATION_WORDS = (
    "published",
    "publication",
    "publish date",
    "start date",
    "cutoff date",
    "veröffentlich",
    "publikations",
    "erscheinungsdatum",
    "startdatum",
    "mindestdatum",
    "stichtag",
)


@dataclass(frozen=True, slots=True)
class DatePromptRecoveryPlan:
    """A deterministic checkpoint migration that has not yet been written."""

    project: Path
    contaminated_cutoff: date
    affected_video_ids: tuple[str, ...]
    old_prompt_sha256: str
    new_prompt: CompiledClassifierPrompt
    cleaned_expansion: TopicExpansion
    updated_state: CrawlProjectState


@dataclass(frozen=True, slots=True)
class DatePromptRecoveryResult:
    """Paths and counts written by an applied recovery."""

    affected_video_count: int
    backup_path: Path
    report_path: Path
    old_prompt_sha256: str
    new_prompt_sha256: str


def plan_date_prompt_recovery(
    project_dir: str | Path,
    *,
    contaminated_cutoff: date,
) -> DatePromptRecoveryPlan:
    """Build a high-precision date-leak repair without changing the project."""

    project = Path(project_dir).expanduser().resolve()
    state = ProjectStateStore(project).load()
    definition = _research_definition(state.classifier_system_prompt)
    cleaned_expansion = _clean_expansion(
        TopicExpansion.model_validate(state.expansion),
        definition=definition,
        contaminated_cutoff=contaminated_cutoff,
    )
    new_prompt = compile_classifier_prompt(
        _brief_from_definition(definition),
        cleaned_expansion,
    )
    affected = _date_contaminated_decision_ids(
        project,
        state,
        contaminated_cutoff=contaminated_cutoff,
    )
    updated = state.model_copy(
        deep=True,
        update={
            "expansion": cleaned_expansion.model_dump(mode="json"),
            "classifier_system_prompt": new_prompt.system_prompt,
            "prompt_sha256": new_prompt.prompt_sha256,
            "classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
            "finalized_video_ids": sorted(set(state.finalized_video_ids) - affected),
            "dispositioned_video_ids": sorted(
                set(state.dispositioned_video_ids) - affected
            ),
            "terminal_video_ids": sorted(set(state.terminal_video_ids) - affected),
            "queued_video_ids": sorted(set(state.queued_video_ids) - affected),
            "last_status": "prepared",
        },
    )
    return DatePromptRecoveryPlan(
        project=project,
        contaminated_cutoff=contaminated_cutoff,
        affected_video_ids=tuple(sorted(affected)),
        old_prompt_sha256=state.prompt_sha256,
        new_prompt=new_prompt,
        cleaned_expansion=cleaned_expansion,
        updated_state=updated,
    )


def apply_date_prompt_recovery(
    plan: DatePromptRecoveryPlan,
) -> DatePromptRecoveryResult:
    """Back up and atomically replace only the mutable crawler checkpoint."""

    store = ProjectStateStore(plan.project)
    current = store.load()
    if current.prompt_sha256 != plan.old_prompt_sha256:
        raise ValueError("crawl_state.json changed after the recovery was planned")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = plan.project / f"crawl_state.before-date-recovery.{timestamp}.json"
    report = plan.project / f"date_prompt_recovery.{timestamp}.json"
    shutil.copy2(store.path, backup)
    store.save(plan.updated_state)
    payload = {
        "applied_at": datetime.now(UTC).isoformat(),
        "contaminated_cutoff": plan.contaminated_cutoff.isoformat(),
        "affected_video_count": len(plan.affected_video_ids),
        "affected_video_ids": list(plan.affected_video_ids),
        "old_prompt_sha256": plan.old_prompt_sha256,
        "new_prompt_sha256": plan.new_prompt.prompt_sha256,
        "classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
        "checkpoint_backup": backup.name,
        "audit_streams_modified": False,
    }
    temporary = report.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report)
    return DatePromptRecoveryResult(
        affected_video_count=len(plan.affected_video_ids),
        backup_path=backup,
        report_path=report,
        old_prompt_sha256=plan.old_prompt_sha256,
        new_prompt_sha256=plan.new_prompt.prompt_sha256,
    )


def _research_definition(system_prompt: str) -> dict[str, Any]:
    try:
        raw = system_prompt.split(_RESEARCH_DEFINITION_MARKER, maxsplit=1)[1]
        payload = json.loads(raw)
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError(
            "classifier prompt has no readable research definition"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("classifier research definition must be a JSON object")
    return payload


def _brief_from_definition(definition: dict[str, Any]) -> TopicBrief:
    return TopicBrief(
        topic_query=str(definition["topic_query"]),
        language=str(definition["requested_language"]),
        research_goal=str(definition["research_goal"]),
        positive_examples=tuple(definition.get("positive_examples_verbatim", ())),
        negative_examples=tuple(definition.get("negative_examples_verbatim", ())),
        must_include=tuple(definition.get("must_include_verbatim", ())),
        edge_case_guidance=tuple(
            definition.get("edge_case_guidance_verbatim", ())
        ),
    )


def _clean_expansion(
    expansion: TopicExpansion,
    *,
    definition: dict[str, Any],
    contaminated_cutoff: date,
) -> TopicExpansion:
    def keep(text: str) -> bool:
        return not _mentions_publication_filter(text, contaminated_cutoff)

    interpretation = expansion.topic_interpretation
    if not keep(interpretation):
        interpretation = str(definition["research_goal"])
    return TopicExpansion(
        topic_interpretation=interpretation,
        inclusion_criteria=tuple(filter(keep, expansion.inclusion_criteria)),
        exclusion_criteria=tuple(filter(keep, expansion.exclusion_criteria)),
        search_queries=expansion.search_queries,
        channel_discovery_queries=expansion.channel_discovery_queries,
        ambiguity_rules=tuple(filter(keep, expansion.ambiguity_rules)),
    )


def _date_contaminated_decision_ids(
    project: Path,
    state: CrawlProjectState,
    *,
    contaminated_cutoff: date,
) -> set[str]:
    decisions = _latest_jsonl_records(project / "relevance_decision.jsonl")
    candidates = _latest_dated_candidate_records(project / "video_candidate.jsonl")
    affected: set[str] = set()
    for video_id, decision in decisions.items():
        if video_id not in state.discovered_videos:
            continue
        if video_id not in state.terminal_video_ids:
            continue
        if decision.get("label") != "irrelevant":
            continue
        if decision.get("decision_point") not in {"video_metadata", "transcript"}:
            continue
        candidate = candidates.get(video_id, {})
        published_at = candidate.get("published_at")
        if not isinstance(published_at, str):
            continue
        published = date.fromisoformat(published_at[:10])
        if published < state.start_date or published >= contaminated_cutoff:
            continue
        criteria = decision.get("criteria", ())
        if not isinstance(criteria, list):
            continue
        if any(
            isinstance(item, dict)
            and item.get("matched") is False
            and isinstance(item.get("criterion"), str)
            and _mentions_publication_filter(
                str(item["criterion"]), contaminated_cutoff
            )
            for item in criteria
        ):
            affected.add(video_id)
    return affected


def _mentions_publication_filter(text: str, cutoff: date) -> bool:
    folded = text.casefold()
    has_publication_word = any(word in folded for word in _PUBLICATION_WORDS)
    has_cutoff = str(cutoff.year) in folded or cutoff.isoformat() in folded
    return has_cutoff or has_publication_word


def _latest_jsonl_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"invalid {path.name} record")
                video_id = payload.get("video_id")
                if not isinstance(video_id, str):
                    raise ValueError(f"invalid {path.name} record")
                latest[video_id] = payload
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot safely read {path.name}") from exc
    return latest


def _latest_dated_candidate_records(path: Path) -> dict[str, dict[str, Any]]:
    """Keep the latest exact-date observation even if a later row has no date."""

    latest: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"invalid {path.name} record")
                video_id = payload.get("video_id")
                published_at = payload.get("published_at")
                if not isinstance(video_id, str):
                    raise ValueError(f"invalid {path.name} record")
                if isinstance(published_at, str):
                    latest[video_id] = payload
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot safely read {path.name}") from exc
    return latest


__all__ = [
    "DatePromptRecoveryPlan",
    "DatePromptRecoveryResult",
    "apply_date_prompt_recovery",
    "plan_date_prompt_recovery",
]
