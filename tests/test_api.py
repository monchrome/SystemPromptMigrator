import pytest
from fastapi.testclient import TestClient

from promptmigrator.api import create_app
from promptmigrator.pipeline import PromptMigrator
from promptmigrator.providers import vendor_for_model
from tests.conftest import FakeProvider


def _fake_provider_for(model: str) -> FakeProvider:
    vendor_for_model(model)  # keep real vendor validation (raises for unsupported)
    return FakeProvider()


@pytest.fixture
def client() -> TestClient:
    migrator = PromptMigrator(provider_for=_fake_provider_for)
    return TestClient(create_app(migrator))


def _upload(content: bytes = b"Summarize the ticket. Think step by step.", name: str = "prompt.txt"):
    return {"prompt_file": (name, content, "text/plain")}


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_model_profiles_endpoint(client: TestClient) -> None:
    response = client.get("/v1/model-profiles")
    assert response.status_code == 200
    families = [p["family"] for p in response.json()]
    assert "claude-4.6+" in families and "gpt" in families


def test_migration_happy_path(client: TestClient) -> None:
    response = client.post(
        "/v1/migrations",
        files=_upload(),
        data={
            "source_model": "gpt-4o",
            "target_model": "claude-opus-4-8",
            "num_candidates": "2",
            "refine": "false",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["migrated_prompt"].startswith("migrated prompt")
    assert body["source_model"] == "gpt-4o"
    assert body["target_model"] == "claude-opus-4-8"
    assert len(body["candidates"]) == 2
    assert body["analysis"]["output_format"]["type"] == "json"


def test_rejects_non_text_extension(client: TestClient) -> None:
    response = client.post(
        "/v1/migrations",
        files=_upload(name="prompt.pdf"),
        data={"source_model": "gpt-4o", "target_model": "claude-opus-4-8"},
    )
    assert response.status_code == 400


def test_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/v1/migrations",
        files=_upload(content=b"   "),
        data={"source_model": "gpt-4o", "target_model": "claude-opus-4-8"},
    )
    assert response.status_code == 400


def test_rejects_invalid_utf8(client: TestClient) -> None:
    response = client.post(
        "/v1/migrations",
        files=_upload(content=b"\xff\xfe\xfa"),
        data={"source_model": "gpt-4o", "target_model": "claude-opus-4-8"},
    )
    assert response.status_code == 400


def test_rejects_same_source_and_target(client: TestClient) -> None:
    response = client.post(
        "/v1/migrations",
        files=_upload(),
        data={"source_model": "claude-opus-4-8", "target_model": "claude-opus-4-8"},
    )
    assert response.status_code == 400


def test_unsupported_target_returns_422(client: TestClient) -> None:
    response = client.post(
        "/v1/migrations",
        files=_upload(),
        data={"source_model": "gpt-4o", "target_model": "gemini-2.5-pro"},
    )
    assert response.status_code == 422
    assert "No execution provider" in response.json()["detail"]
