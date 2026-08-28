from __future__ import annotations

import json
from datetime import date

from yt_searchapi.prompts import (
    CLASSIFIER_PROMPT_VERSION,
    TopicBrief,
    TopicExpansion,
    compile_classifier_prompt,
)
from yt_searchapi.recovery import (
    apply_date_prompt_recovery,
    plan_date_prompt_recovery,
)
from yt_searchapi.state import BudgetState, CrawlProjectState, ProjectStateStore


def _contaminated_project(tmp_path):
    project = tmp_path / "date-leak"
    brief = TopicBrief(
        topic_query="personal finance",
        language="de",
        research_goal="Private Finanzentscheidungen in Deutschland",
        positive_examples=("Ein Haushaltsbudget erklären",),
        negative_examples=("Unternehmensbuchhaltung",),
    )
    expansion = TopicExpansion(
        topic_interpretation="Deutschsprachige private Finanzen ab 2025",
        inclusion_criteria=(
            "Private Finanzentscheidungen erklären",
            "Ab dem 1. Januar 2025 veröffentlicht",
        ),
        exclusion_criteria=("Reine Unternehmensbuchhaltung",),
        search_queries=("private Finanzen 2025",),
        channel_discovery_queries=("Finanzkanal Deutschland",),
        ambiguity_rules=("Unklares Veröffentlichungsdatum ausschließen",),
    )
    prompt = compile_classifier_prompt(brief, expansion)
    discovered = {
        video_id: {
            "video_id": video_id,
            "title": video_id,
            "channel_id": None,
            "channel_title": None,
            "source": "search",
            "source_ref": "query",
            "depth": 0,
        }
        for video_id in ("date-only", "off-topic", "overwritten")
    }
    state = CrawlProjectState(
        run_id=project.name,
        topic_query="personal finance",
        language="de",
        start_date=date(2016, 1, 1),
        gl="de",
        hl="de",
        transcript_excerpt_chars=1000,
        max_depth=0,
        max_queries=1,
        max_search_pages=1,
        max_channel_pages=1,
        expansion=expansion.model_dump(mode="json"),
        classifier_system_prompt=prompt.system_prompt,
        prompt_sha256=prompt.prompt_sha256,
        classifier_prompt_version="relevance-v2",
        planned_queries=[{"text": "private Finanzen 2025", "kind": "seed"}],
        budget=BudgetState(
            max_credits=10,
            transcript_capacity=3,
            discovery_spent=0,
            transcript_spent=0,
        ),
        discovered_videos=discovered,
        evaluated_video_ids=list(discovered),
        finalized_video_ids=list(discovered),
        dispositioned_video_ids=list(discovered),
        terminal_video_ids=list(discovered),
        last_status="completed",
    )
    ProjectStateStore(project).save(state)
    candidates = [
        {"video_id": video_id, "published_at": "2020-01-01T00:00:00Z"}
        for video_id in discovered
    ]
    candidates.append({"video_id": "date-only", "published_at": None})
    decisions = [
        {
            "video_id": "date-only",
            "label": "irrelevant",
            "decision_point": "video_metadata",
            "criteria": [
                {
                    "criterion": "Veröffentlicht vor dem 1. Januar 2025",
                    "matched": False,
                }
            ],
        },
        {
            "video_id": "off-topic",
            "label": "irrelevant",
            "decision_point": "video_metadata",
            "criteria": [{"criterion": "Kein privater Finanzbezug", "matched": False}],
        },
        {
            "video_id": "overwritten",
            "label": "irrelevant",
            "decision_point": "video_metadata",
            "criteria": [
                {
                    "criterion": "Veröffentlicht vor dem 1. Januar 2025",
                    "matched": False,
                }
            ],
        },
        {
            "video_id": "overwritten",
            "label": "irrelevant",
            "decision_point": "video_metadata",
            "criteria": [{"criterion": "Kein privater Finanzbezug", "matched": False}],
        },
    ]
    (project / "video_candidate.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    decision_path = project / "relevance_decision.jsonl"
    decision_path.write_text(
        "".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8"
    )
    return project, decision_path.read_bytes()


def test_recovery_reopens_only_latest_structured_date_failures(tmp_path) -> None:
    project, original_audit = _contaminated_project(tmp_path)

    plan = plan_date_prompt_recovery(
        project, contaminated_cutoff=date(2025, 1, 1)
    )

    assert plan.affected_video_ids == ("date-only",)
    assert "2025" not in plan.new_prompt.system_prompt
    assert "Veröffentlichungsdatum" not in plan.new_prompt.system_prompt
    assert plan.cleaned_expansion.search_queries == ("private Finanzen 2025",)

    result = apply_date_prompt_recovery(plan)
    repaired = ProjectStateStore(project).load()
    assert repaired.classifier_prompt_version == CLASSIFIER_PROMPT_VERSION
    assert repaired.last_status == "prepared"
    assert "date-only" not in repaired.terminal_video_ids
    assert set(repaired.terminal_video_ids) == {"off-topic", "overwritten"}
    assert (project / "relevance_decision.jsonl").read_bytes() == original_audit
    assert result.backup_path.is_file()
    assert result.report_path.is_file()

    second_plan = plan_date_prompt_recovery(
        project, contaminated_cutoff=date(2025, 1, 1)
    )
    assert second_plan.affected_video_ids == ()
