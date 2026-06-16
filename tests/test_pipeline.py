from promptmigrator.pipeline import PromptMigrator
from tests.conftest import FakeProvider


async def test_migrate_selects_highest_scored_candidate(
    migrator: PromptMigrator, fake_provider: FakeProvider
) -> None:
    result = await migrator.migrate(
        prompt="Summarize this ticket. Think step by step. {",
        source_model="gpt-4o",
        target_model="claude-opus-4-8",
        num_candidates=3,
        refine=False,
    )
    # Fake evaluator scores increase per call, so the last candidate wins.
    assert result.selected_candidate_index == 2
    assert result.migrated_prompt == "migrated prompt v3"
    assert result.refined is False
    assert len(result.candidates) == 3
    assert all(c.total_score is not None for c in result.candidates)
    assert result.source_profile.family == "gpt"
    assert result.target_profile.family == "claude-4.6+"


async def test_refinement_replaces_winner_when_scores_improve(
    migrator: PromptMigrator,
) -> None:
    result = await migrator.migrate(
        prompt="Summarize this ticket.",
        source_model="gpt-4o",
        target_model="claude-opus-4-8",
        num_candidates=2,
        refine=True,
    )
    # Refinement evaluation comes later, hence scores >= winner: refined kept.
    assert result.refined is True
    assert result.migrated_prompt == "migrated prompt v3"  # 2 candidates + 1 refine


async def test_pipeline_runs_on_target_model(
    migrator: PromptMigrator, fake_provider: FakeProvider
) -> None:
    await migrator.migrate(
        prompt="p",
        source_model="gpt-4o",
        target_model="claude-opus-4-8",
        num_candidates=1,
        refine=False,
    )
    assert all(call["model"] == "claude-opus-4-8" for call in fake_provider.calls)
    schema_names = [call["schema_name"] for call in fake_provider.calls]
    assert schema_names == ["prompt_analysis", "prompt_proposal", "candidate_evaluation"]


async def test_num_candidates_is_clamped(migrator: PromptMigrator) -> None:
    result = await migrator.migrate(
        prompt="p",
        source_model="gpt-4o",
        target_model="claude-opus-4-8",
        num_candidates=99,
        refine=False,
    )
    assert len(result.candidates) == 5
