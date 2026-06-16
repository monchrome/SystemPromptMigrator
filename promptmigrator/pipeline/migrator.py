"""Pipeline orchestrator: analyze → propose (parallel) → evaluate (parallel) → refine."""

import asyncio
import json
from typing import Callable

from .. import config
from ..knowledge_base import get_profile
from ..models import Candidate, MigrationResponse
from ..providers import provider_for as default_provider_for
from ..providers.base import LLMProvider
from . import analyzer, evaluator, proposer


class PromptMigrator:
    def __init__(self, provider_for: Callable[[str], LLMProvider] | None = None):
        self._provider_for = provider_for or default_provider_for

    async def migrate(
        self,
        *,
        prompt: str,
        source_model: str,
        target_model: str,
        num_candidates: int = config.DEFAULT_NUM_CANDIDATES,
        refine: bool = True,
        notes: str | None = None,
    ) -> MigrationResponse:
        # Resolve the execution provider up front so an unsupported target fails fast.
        provider = self._provider_for(target_model)
        source_profile = get_profile(source_model)
        target_profile = get_profile(target_model)
        num_candidates = max(1, min(num_candidates, config.MAX_NUM_CANDIDATES))
        max_tokens = config.MAX_OUTPUT_TOKENS

        # Stage 1 — structured analysis of the source prompt, by the target model.
        analysis = await analyzer.analyze(
            provider,
            executor_model=target_model,
            source_model=source_model,
            prompt=prompt,
            source_profile=source_profile,
            notes=notes,
            max_tokens=max_tokens,
        )

        # Stage 2 — N diversified candidate rewrites, generated concurrently.
        tips = [
            proposer.PROPOSAL_TIPS[i % len(proposer.PROPOSAL_TIPS)]
            for i in range(num_candidates)
        ]
        proposals_raw = await asyncio.gather(
            *(
                proposer.propose(
                    provider,
                    target_model=target_model,
                    source_model=source_model,
                    prompt=prompt,
                    analysis=analysis,
                    source_profile=source_profile,
                    target_profile=target_profile,
                    tip=tip,
                    notes=notes,
                    max_tokens=max_tokens,
                )
                for tip in tips
            )
        )
        candidates = []
        for tip, raw in zip(tips, proposals_raw):
            data = json.loads(raw)
            candidates.append(
                Candidate(
                    tip=tip,
                    prompt=data["migrated_prompt"],
                    change_log=data["change_log"],
                )
            )

        # Stage 3 — judge every candidate concurrently.
        evaluations = await asyncio.gather(
            *(
                evaluator.evaluate(
                    provider,
                    target_model=target_model,
                    source_model=source_model,
                    original_prompt=prompt,
                    candidate_prompt=candidate.prompt,
                    analysis=analysis,
                    target_profile=target_profile,
                    max_tokens=max_tokens,
                )
                for candidate in candidates
            )
        )
        for candidate, (scores, feedback) in zip(candidates, evaluations):
            candidate.scores = scores
            candidate.total_score = scores.total
            candidate.feedback = feedback

        best_index = max(
            range(len(candidates)), key=lambda i: candidates[i].total_score or 0
        )
        best = candidates[best_index]
        migrated_prompt = best.prompt
        refinement_applied = False

        # Stage 4 — one feedback-driven refinement pass; keep it only if the judge
        # scores it at least as well as the unrefined winner.
        if refine and best.feedback:
            refined_raw = await proposer.refine(
                provider,
                target_model=target_model,
                source_model=source_model,
                candidate_prompt=best.prompt,
                feedback=best.feedback,
                analysis=analysis,
                target_profile=target_profile,
                max_tokens=max_tokens,
            )
            refined = json.loads(refined_raw)
            refined_scores, _ = await evaluator.evaluate(
                provider,
                target_model=target_model,
                source_model=source_model,
                original_prompt=prompt,
                candidate_prompt=refined["migrated_prompt"],
                analysis=analysis,
                target_profile=target_profile,
                max_tokens=max_tokens,
            )
            if refined_scores.total >= (best.total_score or 0):
                migrated_prompt = refined["migrated_prompt"]
                refinement_applied = True

        return MigrationResponse(
            source_model=source_model,
            target_model=target_model,
            migrated_prompt=migrated_prompt,
            selected_candidate_index=best_index,
            refined=refinement_applied,
            analysis=analysis,
            candidates=candidates,
            source_profile=source_profile,
            target_profile=target_profile,
        )
