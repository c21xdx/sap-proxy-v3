import json

from fastapi.testclient import TestClient

from app import main
from app.openai_api import SupportedModel

client = TestClient(main.app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["project"] == "sap-proxy-v3"
    assert "session_ttl_seconds" in body
    assert "session_cached" in body
    assert "session_cache_entries" in body


def test_models_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "")
    monkeypatch.setattr(main.settings, "sap_user", "user")
    monkeypatch.setattr(main.settings, "sap_pass", "pass")
    monkeypatch.setattr(
        main,
        "fetch_supported_models",
        lambda username, password, base_url=None: [SupportedModel(id="anthropic--claude-4.5-sonnet", version="1", owned_by="anthropic")],
    )

    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "anthropic--claude-4.5-sonnet" in ids
    assert "claude-sonnet-4-5" in ids
    assert "claude-4.5-sonnet" in ids


def test_chat_completions_non_stream(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "")
    monkeypatch.setattr(main.settings, "sap_user", "user")
    monkeypatch.setattr(main.settings, "sap_pass", "pass")
    monkeypatch.setattr(
        main,
        "fetch_supported_models",
        lambda username, password, base_url=None: [SupportedModel(id="gpt-5.4", version="1", owned_by="openai")],
    )

    class FakeCompletionResult:
        status_code = 200
        stream_resp = None
        body = '\n'.join([
            'data: {"final_result":{"llm":{"choices":[{"delta":{"content":"OK"}}],"usage":{"prompt_tokens":2,"completion_tokens":1}}}}',
            'data: [DONE]',
        ])

    monkeypatch.setattr(main, "execute_completion_with_password_curl_cffi", lambda *args, **kwargs: FakeCompletionResult())

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "say ok"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "OK"


def test_chat_completions_stream(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "")
    monkeypatch.setattr(main.settings, "sap_user", "user")
    monkeypatch.setattr(main.settings, "sap_pass", "pass")
    monkeypatch.setattr(
        main,
        "fetch_supported_models",
        lambda username, password, base_url=None: [SupportedModel(id="anthropic--claude-4.5-sonnet", version="1", owned_by="anthropic")],
    )

    class FakeCompletionResult:
        status_code = 200
        stream_resp = None
        body = '\n'.join([
            'data: {"final_result":{"llm":{"choices":[{"delta":{"content":"chunk"}}],"usage":{"prompt_tokens":2,"completion_tokens":1}}}}',
            'data: [DONE]',
        ])

    monkeypatch.setattr(main, "execute_completion_with_password_curl_cffi", lambda *args, **kwargs: FakeCompletionResult())

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "anthropic--claude-4.5-sonnet",
            "messages": [{"role": "user", "content": "say ok"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in resp.text


def test_debug_session_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "secret")
    monkeypatch.setattr(
        main.session_cache,
        "snapshot",
        lambda: [{"username": "u1", "base_url": "https://a.example.com", "age_seconds": 1, "expires_in_seconds": 1499, "final_url": "https://a.example.com/aic/index.html"}],
    )

    resp = client.get("/debug/session", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_ttl_seconds"] == main.settings.session_ttl_seconds
    assert len(body["entries"]) == 1
    assert body["entries"][0]["username"] == "**"


def test_debug_session_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "secret")
    resp = client.get("/debug/session")
    assert resp.status_code == 401


def test_models_endpoint_surfaces_metadata_error(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "")
    monkeypatch.setattr(main, 'fetch_supported_models', lambda *args, **kwargs: (_ for _ in ()).throw(main.SAPMetadataError('bad metadata')))
    resp = client.get('/v1/models')
    assert resp.status_code == 502
    assert 'sap metadata error' in resp.json()['detail']


def test_chat_completions_surfaces_completion_error(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, 'api_key', '')
    monkeypatch.setattr(main.settings, 'sap_user', 'user')
    monkeypatch.setattr(main.settings, 'sap_pass', 'pass')
    monkeypatch.setattr(
        main,
        'fetch_supported_models',
        lambda username, password, base_url=None: [SupportedModel(id='gpt-5.4', version='1', owned_by='openai')],
    )
    monkeypatch.setattr(
        main,
        'execute_completion_with_password_curl_cffi',
        lambda *args, **kwargs: (_ for _ in ()).throw(main.SAPCompletionError('bad completion')),
    )

    resp = client.post(
        '/v1/chat/completions',
        json={
            'model': 'gpt-5.4',
            'messages': [{'role': 'user', 'content': 'say ok'}],
            'stream': False,
        },
    )
    assert resp.status_code == 502
    assert 'sap completion error' in resp.json()['detail']


def test_request_id_header_is_set() -> None:
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.headers['x-request-id'].startswith('req_')


def test_request_id_header_is_preserved() -> None:
    resp = client.get('/health', headers={'X-Request-ID': 'custom-123'})
    assert resp.status_code == 200
    assert resp.headers['x-request-id'] == 'custom-123'


def test_chat_completions_stream_sets_model_header(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "api_key", "")
    monkeypatch.setattr(main.settings, "sap_user", "user")
    monkeypatch.setattr(main.settings, "sap_pass", "pass")
    monkeypatch.setattr(
        main,
        "fetch_supported_models",
        lambda username, password, base_url=None: [SupportedModel(id="anthropic--claude-4.5-sonnet", version="1", owned_by="anthropic")],
    )

    class FakeCompletionResult:
        status_code = 200
        stream_resp = None
        body = '\n'.join([
            'data: {"final_result":{"llm":{"choices":[{"delta":{"content":"chunk"}}],"usage":{"prompt_tokens":2,"completion_tokens":1}}}}',
            'data: [DONE]',
        ])

    monkeypatch.setattr(main, "execute_completion_with_password_curl_cffi", lambda *args, **kwargs: FakeCompletionResult())

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "anthropic--claude-4.5-sonnet",
            "messages": [{"role": "user", "content": "say ok"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["x-model"] == "anthropic--claude-4.5-sonnet"


def test_chat_completions_tools_non_stream(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, 'api_key', '')
    monkeypatch.setattr(main.settings, 'sap_user', 'user')
    monkeypatch.setattr(main.settings, 'sap_pass', 'pass')
    monkeypatch.setattr(
        main,
        'fetch_supported_models',
        lambda username, password, base_url=None: [SupportedModel(id='anthropic--claude-4.6-sonnet', version='1', owned_by='anthropic')],
    )

    class FakeCompletionResult:
        status_code = 200
        stream_resp = None
        body = json.dumps({
            'final_result': {
                'llm': {
                    'choices': [
                        {'message': {'role': 'assistant', 'content': '<function_call>\nget_weather\n{"city":"New York"}\n</function_call>'}}
                    ],
                    'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
                }
            }
        })

    monkeypatch.setattr(main, 'execute_completion_with_password_curl_cffi', lambda *args, **kwargs: FakeCompletionResult())

    resp = client.post(
        '/v1/chat/completions',
        json={
            'model': 'anthropic--claude-4.6-sonnet',
            'messages': [{'role': 'user', 'content': 'Use weather tool'}],
            'tools': [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get weather', 'parameters': {'type': 'object'}}}],
            'stream': False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body['choices'][0]['finish_reason'] == 'tool_calls'
    assert body['choices'][0]['message']['tool_calls'][0]['function']['name'] == 'get_weather'


def test_chat_completions_unsupported_model_returns_404(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, 'api_key', '')
    monkeypatch.setattr(main.settings, 'sap_user', 'user')
    monkeypatch.setattr(main.settings, 'sap_pass', 'pass')

    resp = client.post(
        '/v1/chat/completions',
        json={
            'model': 'totally-made-up-model',
            'messages': [{'role': 'user', 'content': 'test'}],
            'stream': False,
        },
    )
    assert resp.status_code == 404
    assert 'unsupported model' in resp.json()['detail']
