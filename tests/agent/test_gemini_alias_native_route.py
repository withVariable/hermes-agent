"""Provider aliases preserve the native-vs-compatible client boundary."""
from types import SimpleNamespace

from openai import OpenAI

from agent.agent_runtime_helpers import create_openai_client
from agent.gemini_native_adapter import GeminiNativeClient
from providers import get_provider_profile


def test_gemini_aliases_preserve_native_endpoint_selection():
    profile = get_provider_profile("gemini")
    for provider in (profile.name, *profile.aliases):
        for base_url, expected in (
            ("https://generativelanguage.googleapis.com/v1beta", GeminiNativeClient),
            ("https://example.invalid/v1", OpenAI),
        ):
            agent = SimpleNamespace(
                provider=provider, model="gemini-flash-latest",
                _client_log_context=lambda: "test",
                _build_keepalive_http_client=lambda *a, **k: None,
            )
            client = create_openai_client(
                agent, {"api_key": "test-key", "base_url": base_url},
                reason="fallback", shared=False,
            )
            try:
                assert isinstance(client, expected), (provider, base_url, type(client))
            finally:
                client.close()


def test_configured_gemini_base_url_proxy_is_native(monkeypatch):
    """An explicit GEMINI_BASE_URL (a native-Gemini proxy root) selects the native client even
    though its host is not Google's; an unrelated base or an /openai proxy stays compatible."""
    from agent.gemini_native_adapter import is_native_gemini_base_url

    monkeypatch.setenv("GEMINI_BASE_URL", "https://gateway.example/providers/vertex")
    assert is_native_gemini_base_url("https://gateway.example/providers/vertex")
    assert is_native_gemini_base_url("https://gateway.example/providers/vertex/v1beta/")
    assert not is_native_gemini_base_url("https://gateway.example/providers/openrouter/v1")
    assert not is_native_gemini_base_url("https://gateway.example/providers/vertex/openai")

    monkeypatch.setenv("GEMINI_BASE_URL", "https://gateway.example/providers/vertex/openai")
    assert not is_native_gemini_base_url("https://gateway.example/providers/vertex/openai")

    monkeypatch.delenv("GEMINI_BASE_URL")
    assert not is_native_gemini_base_url("https://gateway.example/providers/vertex")


def test_configured_gemini_proxy_builds_native_client(monkeypatch):
    monkeypatch.setenv("GEMINI_BASE_URL", "https://gateway.example/providers/vertex")
    agent = SimpleNamespace(
        provider="gemini", model="gemini-4-hydrogen",
        _client_log_context=lambda: "test",
        _build_keepalive_http_client=lambda *a, **k: None,
    )
    client = create_openai_client(
        agent, {"api_key": "gateway-token", "base_url": "https://gateway.example/providers/vertex"},
        reason="primary", shared=False,
    )
    try:
        assert isinstance(client, GeminiNativeClient)
        assert client.base_url == "https://gateway.example/providers/vertex/v1beta"
    finally:
        client.close()


def test_managed_gemini_proxy_config_resolves_to_the_proxy(monkeypatch):
    """A managed config that names the gemini provider with a proxy base URL and the
    proxy's token resolves to exactly that endpoint and key (no Google default)."""
    from hermes_cli import runtime_provider as rp

    gateway = "https://gateway.example/providers/vertex"
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "gemini", "default": "google/gemini-4-hydrogen",
        "base_url": gateway, "api_key": "gateway-token", "api_mode": "chat_completions",
    })
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gateway-token")
    monkeypatch.setenv("GEMINI_BASE_URL", gateway)

    resolved = rp.resolve_runtime_provider(requested="gemini")

    assert resolved["provider"] == "gemini"
    assert resolved["api_key"] == "gateway-token"
    assert resolved["base_url"].rstrip("/") in {gateway, f"{gateway}/v1beta"}
    assert resolved["api_mode"] == "chat_completions"
