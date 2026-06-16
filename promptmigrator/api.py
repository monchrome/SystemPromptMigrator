"""FastAPI surface for the prompt-migration pipeline."""

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from . import config
from .knowledge_base import ModelProfile, all_profiles
from .models import MigrationResponse
from .pipeline import PromptMigrator
from .providers.base import ProviderError


def create_app(migrator: PromptMigrator | None = None) -> FastAPI:
    app = FastAPI(
        title="PromptMigrator",
        description=(
            "Migrates a prompt from a source LLM to a target LLM. The target model "
            "itself rewrites the prompt using a MIPROv2-style propose/evaluate/refine "
            "loop grounded in a cross-model trait knowledge base."
        ),
        version="0.1.0",
    )
    app.state.migrator = migrator or PromptMigrator()

    @app.exception_handler(ProviderError)
    async def provider_error_handler(_: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc), "retryable": exc.retryable},
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/model-profiles", response_model=list[ModelProfile])
    async def model_profiles() -> list[ModelProfile]:
        """The cross-model trait knowledge base used to calibrate migrations."""
        return all_profiles()

    @app.post("/v1/migrations", response_model=MigrationResponse)
    async def create_migration(
        prompt_file: UploadFile = File(..., description="The source prompt as a .txt upload"),
        source_model: str = Form(..., description="Model the prompt was written for"),
        target_model: str = Form(..., description="Model to migrate the prompt to"),
        num_candidates: int = Form(
            config.DEFAULT_NUM_CANDIDATES,
            ge=1,
            le=config.MAX_NUM_CANDIDATES,
            description="Diversified candidate rewrites to generate (MIPROv2-style)",
        ),
        refine: bool = Form(True, description="Run a feedback-driven refinement pass"),
        notes: str | None = Form(
            None, description="Optional operator notes (task context, eval criteria)"
        ),
    ) -> MigrationResponse:
        if prompt_file.filename and not prompt_file.filename.lower().endswith(
            (".txt", ".md", ".prompt")
        ):
            raise HTTPException(
                status_code=400,
                detail="prompt_file must be a plain-text file (.txt, .md, or .prompt).",
            )
        raw = await prompt_file.read()
        if len(raw) > config.MAX_PROMPT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"prompt_file exceeds {config.MAX_PROMPT_BYTES} bytes.",
            )
        try:
            prompt = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=400, detail="prompt_file is not valid UTF-8 text."
            ) from e
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="prompt_file is empty.")
        if source_model.strip().lower() == target_model.strip().lower():
            raise HTTPException(
                status_code=400, detail="source_model and target_model are identical."
            )

        return await app.state.migrator.migrate(
            prompt=prompt,
            source_model=source_model.strip(),
            target_model=target_model.strip(),
            num_candidates=num_candidates,
            refine=refine,
            notes=notes,
        )

    return app


app = create_app()
