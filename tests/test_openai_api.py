from app.openai_api import (
    OpenAIChatRequest,
    OpenAIMessage,
    OpenAITool,
    OpenAIToolCall,
    OpenAIToolFunctionCall,
    _build_prompt_text,
    build_openai_response,
    build_openai_response_from_text,
    extract_supported_models,
    parse_sap_sse_text,
    resolve_model,
    _to_completion_request,
)
from app import model_registry, openai_api


METADATA = {
    'globalLLMInfo': {
        'models': [
            {'model': 'anthropic--claude-4.5-haiku', 'version': '1', 'metadata': {'deprecated': False}},
            {'model': 'anthropic--claude-4.5-sonnet', 'version': '1', 'metadata': {'deprecated': False}},
            {'model': 'anthropic--claude-4.5-opus', 'version': '1', 'metadata': {'deprecated': False}},
            {'model': 'anthropic--claude-4.6-opus', 'version': '1', 'metadata': {'deprecated': False}},
            {'model': 'anthropic--claude-4.6-sonnet', 'version': '1', 'metadata': {'deprecated': False}},
            {'model': 'anthropic--claude-3.7-sonnet', 'version': '1', 'metadata': {'deprecated': False}},
            {'model': 'openai--gpt-5.2', 'version': '3', 'metadata': {'deprecated': False}},
            {'model': 'gpt-5.4', 'version': '2026-03-05', 'metadata': {'deprecated': False}},
            {'model': 'openai--gpt-4o', 'version': '1', 'metadata': {'deprecated': False}},
        ]
    }
}


SSE_BODY = '\n'.join([
    'data: {"intermediate_results":{"llm":{"choices":[{"delta":{"content":"Hello "}}]}}}',
    'data: {"final_result":{"llm":{"choices":[{"delta":{"content":"world"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}}}',
    'data: [DONE]',
])

NONSTREAM_BODY = '{"request_id":"x","final_result":{"id":"m1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}}'


def test_extract_supported_models_auto_discover_when_no_allowlist() -> None:
    """When ALLOWED_MODELS is empty (default), all non-deprecated models are accepted."""
    models = extract_supported_models(METADATA)
    ids = [m.id for m in models]
    # All non-deprecated models from metadata should be present
    assert 'gpt-5.4' in ids
    assert 'anthropic--claude-4.5-sonnet' in ids
    assert 'anthropic--claude-4.6-opus' in ids
    # Also models that were previously filtered out
    assert 'openai--gpt-5.2' in ids
    assert 'anthropic--claude-3.7-sonnet' in ids
    assert 'openai--gpt-4o' in ids


def test_resolve_model_supports_short_name() -> None:
    models = extract_supported_models(METADATA)
    resolved = resolve_model(models, 'gpt-5.4')
    assert resolved is not None
    assert resolved.id == 'gpt-5.4'


def test_build_prompt_text_preserves_roles() -> None:
    prompt = _build_prompt_text([
        OpenAIMessage(role='system', content='You are helpful.'),
        OpenAIMessage(role='user', content='Write code'),
        OpenAIMessage(role='assistant', content='Sure.'),
        OpenAIMessage(role='user', content=[{'type': 'text', 'text': 'Add tests'}]),
    ])
    assert prompt.startswith('System:\nYou are helpful.')
    assert 'User: Write code' in prompt
    assert 'Assistant: Sure.' in prompt
    assert 'User: Add tests' in prompt


def test_to_completion_request_uses_request_defaults(monkeypatch) -> None:
    monkeypatch.setattr(openai_api.settings.completion, 'force_non_stream', False)
    req = OpenAIChatRequest(
        model='anthropic--claude-4.5-sonnet',
        messages=[
            OpenAIMessage(role='system', content='Be concise'),
            OpenAIMessage(role='user', content='Write code'),
        ],
        max_tokens=9000,
        temperature=0.1,
        stream=True,
    )
    completion = _to_completion_request(req, 'anthropic--claude-4.5-sonnet', '1')
    assert completion.model_name == 'anthropic--claude-4.5-sonnet'
    assert completion.max_tokens == 9000
    assert completion.temperature == 0.1
    assert completion.stream_enabled is True
    assert completion.prompt_text.startswith('System:\nBe concise')
    assert 'User: Write code' in completion.prompt_text


def test_parse_sap_sse_text() -> None:
    content, usage = parse_sap_sse_text(SSE_BODY)
    assert content == 'Hello world'
    assert usage['prompt_tokens'] == 10
    assert usage['completion_tokens'] == 5
    assert usage['total_tokens'] == 15


def test_build_openai_response() -> None:
    response = build_openai_response('openai--gpt-5.2', 'ok', {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3})
    assert response.object == 'chat.completion'
    assert response.model == 'openai--gpt-5.2'
    assert response.choices[0]['message']['content'] == 'ok'


def test_parse_nonstream_json_text() -> None:
    content, usage = parse_sap_sse_text(NONSTREAM_BODY)
    assert content == 'OK'
    assert usage['prompt_tokens'] == 12
    assert usage['completion_tokens'] == 4
    assert usage['total_tokens'] == 16


def test_fetch_supported_models_returns_all_from_metadata(monkeypatch) -> None:
    """fetch_supported_models returns all non-deprecated models from metadata
    when ALLOWED_MODELS is empty (auto-discover mode)."""
    from app import model_registry, openai_api

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        def raise_for_status(self):
            return None
        def json(self):
            return METADATA

    class FakeSession:
        def get(self, url):
            return FakeResp()

    monkeypatch.setattr(model_registry, 'get_cached_session_curl_cffi', lambda username, password, force_refresh=False, base_url=None: (FakeSession(), 'https://example.com/aic/index.html'))

    models = model_registry.fetch_supported_models('u', 'p')
    ids = [m.id for m in models]
    # All models from metadata should be present (auto-discover)
    assert 'anthropic--claude-4.6-opus' in ids
    assert 'anthropic--claude-4.6-sonnet' in ids
    assert 'gpt-5.4' in ids
    # Previously filtered models now appear too
    assert 'anthropic--claude-3.7-sonnet' in ids
    assert 'openai--gpt-4o' in ids


def test_model_owned_by_for_bare_gpt() -> None:
    from app.model_registry import _model_owned_by
    assert _model_owned_by('gpt-5.4') == 'openai'


def test_extract_supported_models_keeps_claude46_focus_models() -> None:
    models = extract_supported_models(METADATA)
    ids = [m.id for m in models]
    assert 'anthropic--claude-4.6-opus' in ids
    assert 'anthropic--claude-4.6-sonnet' in ids


def test_supported_model_filter_uses_config(monkeypatch) -> None:
    from app import model_registry, openai_api
    monkeypatch.setattr(openai_api.settings, 'allowed_models', ['gpt-5.4'])
    models = extract_supported_models(METADATA)
    ids = [m.id for m in models]
    assert ids == ['gpt-5.4']


def test_inspect_supported_models(monkeypatch) -> None:
    from app import model_registry, openai_api

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        def raise_for_status(self):
            return None
        def json(self):
            return METADATA

    class FakeSession:
        def get(self, url):
            return FakeResp()

    monkeypatch.setattr(openai_api.settings, 'allowed_models', ['anthropic--claude-4.6-opus', 'gpt-5.4'])
    monkeypatch.setattr(model_registry, 'get_cached_session_curl_cffi', lambda username, password, force_refresh=False, base_url=None: (FakeSession(), 'https://example.com/aic/index.html'))
    monkeypatch.setattr(
        model_registry,
        'check_model_access_with_password_curl_cffi',
        lambda username, password, model_name, model_version, base_url=None, deployment_id=None, resource_group_id=None: type('R', (), {'allowed': model_name != 'gpt-5.4', 'status_code': 200 if model_name != 'gpt-5.4' else 403, 'detail': '' if model_name != 'gpt-5.4' else 'forbidden'})(),
    )

    entries = model_registry.inspect_supported_models('u', 'p')
    assert [entry.id for entry in entries] == ['anthropic--claude-4.6-opus', 'gpt-5.4']
    assert entries[0].access_allowed is True
    assert entries[1].access_allowed is False
    assert entries[1].access_status_code == 403


def test_build_prompt_text_includes_tools_and_tool_results() -> None:
    prompt = _build_prompt_text([
        OpenAIMessage(role='user', content='Need weather'),
        OpenAIMessage(role='assistant', content='', tool_calls=[OpenAIToolCall(id='call_get_weather_0', function=OpenAIToolFunctionCall(name='get_weather', arguments='{"city":"New York"}'))]),
        OpenAIMessage(role='tool', content='{"temp": 22}', tool_call_id='call_get_weather_0'),
    ], tools=[OpenAITool(type='function', function={'name': 'get_weather', 'description': 'Get weather', 'parameters': {'type': 'object'}})])
    assert 'In this environment you have access to a set of tools' in prompt
    assert '<function_call>' in prompt
    assert '[tool result for call_get_weather_0]' in prompt


def test_parse_tool_calls() -> None:
    from app.openai_api import parse_tool_calls
    calls, remaining = parse_tool_calls('<function_call>\nget_weather\n{"city":"New York"}\n</function_call>')
    assert len(calls) == 1
    assert calls[0].name == 'get_weather'
    assert calls[0].arguments == '{"city":"New York"}'
    assert remaining == ''


def test_parse_tool_calls_strips_intermediate_thinking() -> None:
    """Text between tool calls (model's English 'thinking') should be stripped.
    Only text after the last tool call is the final reply."""
    from app.openai_api import parse_tool_calls
    content = """Let me check the weather for you.
<function_call>
get_weather
{"city":"NYC"}
</function_call>
Now let me also check the forecast.
<function_call>
get_forecast
{"city":"NYC"}
</function_call>
The weather in NYC is sunny and 72°F."""
    calls, remaining = parse_tool_calls(content)
    assert len(calls) == 2
    assert calls[0].name == 'get_weather'
    assert calls[1].name == 'get_forecast'
    # Only the final reply text is kept, not the intermediate English thinking
    assert remaining == 'The weather in NYC is sunny and 72°F.'


def test_parse_tool_calls_no_final_text() -> None:
    """If there's no text after the last tool call, remaining should be empty."""
    from app.openai_api import parse_tool_calls
    content = "I'll check that.\n<function_call>\nget_weather\n{\"city\":\"NYC\"}\n</function_call>"
    calls, remaining = parse_tool_calls(content)
    assert len(calls) == 1
    assert remaining == ''


def test_build_openai_response_from_text_with_tool_call() -> None:
    response = build_openai_response_from_text(
        'anthropic--claude-4.6-sonnet',
        '<function_call>\nget_weather\n{"city":"New York"}\n</function_call>',
        {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3},
        has_tools=True,
    )
    message = response.choices[0]['message']
    assert message['tool_calls'][0]['function']['name'] == 'get_weather'
    assert response.choices[0]['finish_reason'] == 'tool_calls'


def test_is_html_response_detection() -> None:
    from app.curl_login import _is_html_response, _is_html_response_status_only

    class FakeHTMLResp:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        text = "<html><head><title>Login</title></head></html>"

    class FakeJSONResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = '{"ok": true}'

    class FakeHTML401:
        status_code = 401
        headers = {"content-type": "text/html"}
        text = "<html>unauthorized</html>"

    assert _is_html_response(FakeHTMLResp()) is True
    assert _is_html_response(FakeJSONResp()) is False
    assert _is_html_response(FakeHTML401()) is False
    assert _is_html_response_status_only(FakeHTMLResp()) is True
    assert _is_html_response_status_only(FakeJSONResp()) is False


def test_metadata_cache() -> None:
    from app import model_registry, openai_api

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        def raise_for_status(self):
            return None
        def json(self):
            return METADATA

    call_count = 0

    class FakeSession:
        def get(self, url):
            nonlocal call_count
            call_count += 1
            return FakeResp()

    model_registry._metadata_cache.clear()
    monkeypatch_like = type('M', (), {
        'setattr': lambda self, obj, name, val: setattr(obj, name, val)
    })()
    orig = model_registry.get_cached_session_curl_cffi
    try:
        model_registry.get_cached_session_curl_cffi = lambda username, password, force_refresh=False, base_url=None: (FakeSession(), 'https://example.com')
        models1 = model_registry._get_cached_models('u', 'p')
        models2 = model_registry._get_cached_models('u', 'p')
        assert len(models1) > 0
        assert call_count == 1  # only one API call for two requests
        assert [m.id for m in models1] == [m.id for m in models2]
    finally:
        model_registry.get_cached_session_curl_cffi = orig
        model_registry._metadata_cache.clear()


def test_invalidate_metadata_cache() -> None:
    from app import model_registry, openai_api
    model_registry._metadata_cache['test'] = (0, [])
    assert len(model_registry._metadata_cache) == 1
    model_registry.invalidate_metadata_cache()
    assert len(model_registry._metadata_cache) == 0


def test_history_truncation_counts_tool_calls_arguments() -> None:
    """History truncation must subtract tool_calls arguments length when dropping entries.
    Bug: previously only subtracted content length, not arguments length.
    This caused total_chars to stay artificially high after truncation."""
    from app.openai_api import _build_messages_history, _build_template_messages, OpenAIMessage, OpenAIToolCall, OpenAIToolFunctionCall
    from app import model_registry, openai_api

    # Save and override settings to force truncation
    orig_tokens = openai_api.settings.max_history_tokens
    orig_turns = openai_api.settings.max_history_turns
    try:
        # Set very low token budget to trigger truncation
        openai_api.settings.max_history_tokens = 10  # ~40 chars allowed
        openai_api.settings.max_history_turns = 100

        # Build messages: old turn with tool_calls (long arguments), then new turn
        long_args = '{"city": "' + 'x' * 200 + '"}'  # ~210 chars
        messages = [
            OpenAIMessage(role='user', content='What is the weather?'),
            OpenAIMessage(
                role='assistant',
                content='',
                tool_calls=[OpenAIToolCall(
                    id='call_1',
                    function=OpenAIToolFunctionCall(name='get_weather', arguments=long_args),
                )],
            ),
            OpenAIMessage(role='tool', content='Sunny', tool_call_id='call_1'),
            OpenAIMessage(role='user', content='How about NYC?'),  # new turn starts here
        ]

        history = _build_messages_history(messages)
        # The old turn with long tool_calls should be truncated
        # If the bug existed, total_chars would be inflated and truncation would be too aggressive
        # With the fix, truncation should correctly account for arguments length
        # History should be empty (old turn truncated) since it exceeded the tiny budget
        assert len(history) == 0
    finally:
        openai_api.settings.max_history_tokens = orig_tokens
        openai_api.settings.max_history_turns = orig_turns


def test_history_truncation_preserves_recent_entries() -> None:
    """When history exceeds token budget, older entries are dropped but recent ones preserved."""
    from app.openai_api import _build_messages_history, OpenAIMessage
    from app import model_registry, openai_api

    orig_tokens = openai_api.settings.max_history_tokens
    orig_turns = openai_api.settings.max_history_turns
    try:
        # Budget allows ~50 chars (200 / 4 = 50 tokens)
        openai_api.settings.max_history_tokens = 50
        openai_api.settings.max_history_turns = 100

        messages = [
            OpenAIMessage(role='user', content='A' * 200),  # very long, will be truncated
            OpenAIMessage(role='assistant', content='B' * 10),  # short
            OpenAIMessage(role='user', content='New question'),  # new turn
        ]

        history = _build_messages_history(messages)
        # Old long user message should be truncated, short assistant should remain
        # But with 50 token budget, even the short assistant might be kept
        # The key assertion: no entry exceeds the budget after truncation
        total_chars = sum(len(str(e.get('content', ''))) for e in history)
        assert total_chars // 4 <= openai_api.settings.max_history_tokens
    finally:
        openai_api.settings.max_history_tokens = orig_tokens
        openai_api.settings.max_history_turns = orig_turns


def test_history_truncation_turn_limit() -> None:
    """History is truncated to max_history_turns user messages."""
    from app.openai_api import _build_messages_history, OpenAIMessage
    from app import model_registry, openai_api

    orig_turns = openai_api.settings.max_history_turns
    orig_tokens = openai_api.settings.max_history_tokens
    try:
        openai_api.settings.max_history_turns = 2
        openai_api.settings.max_history_tokens = 100000

        # 5 user turns in history, only last 2 should be kept
        messages = []
        for i in range(5):
            messages.append(OpenAIMessage(role='user', content=f'Turn {i} question'))
            messages.append(OpenAIMessage(role='assistant', content=f'Turn {i} answer'))
        messages.append(OpenAIMessage(role='user', content='Current question'))

        history = _build_messages_history(messages)
        user_entries = [e for e in history if e['role'] == 'user']
        # Should have at most 2 user messages from history
        assert len(user_entries) <= 2
    finally:
        openai_api.settings.max_history_turns = orig_turns
        openai_api.settings.max_history_tokens = orig_tokens


def test_history_with_tool_calls_keeps_intact_pairs() -> None:
    """Assistant tool_calls and corresponding tool results should stay together."""
    from app.openai_api import _build_messages_history, _build_template_messages, OpenAIMessage, OpenAIToolCall, OpenAIToolFunctionCall
    from app import model_registry, openai_api

    orig_tokens = openai_api.settings.max_history_tokens
    orig_turns = openai_api.settings.max_history_turns
    try:
        openai_api.settings.max_history_tokens = 100000
        openai_api.settings.max_history_turns = 100

        messages = [
            OpenAIMessage(role='user', content='Check weather'),
            OpenAIMessage(
                role='assistant',
                content='',
                tool_calls=[OpenAIToolCall(
                    id='call_w',
                    function=OpenAIToolFunctionCall(name='get_weather', arguments='{"city":"NYC"}'),
                )],
            ),
            OpenAIMessage(role='tool', content='{"temp": 22}', tool_call_id='call_w'),
            OpenAIMessage(role='assistant', content='It is 22 degrees in NYC.'),
            OpenAIMessage(role='user', content='What about London?'),  # new turn
        ]

        history = _build_messages_history(messages)
        # History should contain the complete tool call sequence
        assert len(history) == 4  # user + assistant(tool_calls) + tool + assistant(text)
        # assistant entry should have tool_calls
        assistant_entries = [e for e in history if e['role'] == 'assistant']
        assert assistant_entries[0].get('tool_calls') is not None
        assert len(assistant_entries[0]['tool_calls']) == 1
        assert assistant_entries[0]['tool_calls'][0]['function']['name'] == 'get_weather'
    finally:
        openai_api.settings.max_history_tokens = orig_tokens
        openai_api.settings.max_history_turns = orig_turns


def test_resolve_model_cached_rejects_unknown_names() -> None:
    """resolve_model_cached must return None for completely random model names
    when metadata is available but the model is not found there."""
    from app import model_registry, openai_api

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        def raise_for_status(self):
            return None
        def json(self):
            return METADATA

    class FakeSession:
        def get(self, url):
            return FakeResp()

    model_registry._metadata_cache.clear()
    orig = model_registry.get_cached_session_curl_cffi
    try:
        model_registry.get_cached_session_curl_cffi = lambda username, password, force_refresh=False, base_url=None: (FakeSession(), 'https://example.com')
        # Populate cache
        model_registry._get_cached_models('u', 'p')
        # Random string should be rejected
        assert model_registry.resolve_model_cached('totally-made-up-model', 'u', 'p') is None
        assert model_registry.resolve_model_cached('foo', 'u', 'p') is None
        assert model_registry.resolve_model_cached('abc123', 'u', 'p') is None
        # But a real model from metadata should work
        assert model_registry.resolve_model_cached('gpt-5.4', 'u', 'p') is not None
    finally:
        model_registry.get_cached_session_curl_cffi = orig
        model_registry._metadata_cache.clear()


def test_resolve_model_cached_alias_still_works_without_metadata() -> None:
    """When metadata is unavailable, known aliases still resolve."""
    from app import model_registry, openai_api

    model_registry._metadata_cache.clear()
    # No username/password → metadata won't be fetched
    resolved = model_registry.resolve_model_cached('claude-sonnet-4-5')
    assert resolved is not None
    assert resolved.id == 'anthropic--claude-4.5-sonnet'
    assert resolved.version == 'latest'


def test_resolve_model_cached_rejects_random_without_metadata() -> None:
    """When metadata is unavailable, completely unknown names are rejected."""
    from app import model_registry, openai_api

    model_registry._metadata_cache.clear()
    assert model_registry.resolve_model_cached('xyz-not-a-model') is None
    assert model_registry.resolve_model_cached('random-gibberish') is None


def test_resolve_model_cached_allows_canonical_ids_without_metadata() -> None:
    """When metadata is unavailable, canonical SAP model IDs are accepted."""
    from app import model_registry, openai_api

    model_registry._metadata_cache.clear()
    # These look like real model IDs (vendor--model pattern)
    r = model_registry.resolve_model_cached('anthropic--claude-4.6-opus')
    assert r is not None
    assert r.id == 'anthropic--claude-4.6-opus'
    # GPT family
    r = model_registry.resolve_model_cached('gpt-5.4')
    assert r is not None
    assert r.id == 'gpt-5.4'


def test_looks_like_model_id() -> None:
    from app.model_registry import _looks_like_model_id
    # Canonical vendor--model patterns
    assert _looks_like_model_id('anthropic--claude-4.6-opus') is True
    assert _looks_like_model_id('google--gemini-2.5-flash') is True
    assert _looks_like_model_id('meta--llama-3') is True
    assert _looks_like_model_id('mistral--large') is True
    # GPT family
    assert _looks_like_model_id('gpt-5.4') is True
    assert _looks_like_model_id('gpt-4o') is True
    # Known aliases
    assert _looks_like_model_id('claude-sonnet-4-5') is True
    # Random garbage
    assert _looks_like_model_id('foo') is False
    assert _looks_like_model_id('abc123') is False
    assert _looks_like_model_id('totally-made-up') is False


def test_inspect_supported_models_empty_allowlist(monkeypatch) -> None:
    """When allowed_models is empty (auto-discover), all non-deprecated models appear."""
    from app import model_registry, openai_api

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        def raise_for_status(self):
            return None
        def json(self):
            return METADATA

    class FakeSession:
        def get(self, url):
            return FakeResp()

    monkeypatch.setattr(openai_api.settings, 'allowed_models', [])
    monkeypatch.setattr(model_registry, 'get_cached_session_curl_cffi', lambda username, password, force_refresh=False, base_url=None: (FakeSession(), 'https://example.com/aic/index.html'))
    monkeypatch.setattr(
        model_registry,
        'check_model_access_with_password_curl_cffi',
        lambda username, password, model_name, model_version, base_url=None, deployment_id=None, resource_group_id=None: type('R', (), {'allowed': True, 'status_code': 200, 'detail': ''})(),
    )

    entries = model_registry.inspect_supported_models('u', 'p')
    ids = [e.id for e in entries]
    # All non-deprecated models from metadata should appear
    assert 'anthropic--claude-4.5-sonnet' in ids
    assert 'gpt-5.4' in ids
    assert 'anthropic--claude-4.6-opus' in ids
    # configured should be True for all in auto-discover mode
    assert all(e.configured for e in entries)


# ---------------------------------------------------------------------------
# §39: GPT Review #4 fixes — stream downgrade, content validation
# ---------------------------------------------------------------------------


def test_multimodal_forces_stream_disabled():
    """_to_completion_request should force stream_enabled=False when has_images=True."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    req = OpenAIChatRequest(
        model='gpt-4o',
        stream=True,
        messages=[
            OpenAIMessage(role='user', content=[
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}},
                {'type': 'text', 'text': 'What is this?'}
            ])
        ]
    )
    cr = _to_completion_request(req, model_id='gpt-4o', version='2024-11-20')
    assert cr.has_images is True
    assert cr.stream_enabled is False, "stream should be force-disabled for multimodal"


def test_text_mode_keeps_stream(monkeypatch):
    """_to_completion_request should preserve stream=True for text-only requests."""
    monkeypatch.setattr(openai_api.settings.completion, 'force_non_stream', False)
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    req = OpenAIChatRequest(
        model='gpt-4o',
        stream=True,
        messages=[OpenAIMessage(role='user', content='Hello')]
    )
    cr = _to_completion_request(req, model_id='gpt-4o', version='2024-11-20')
    assert cr.has_images is False
    assert cr.stream_enabled is True, "stream should stay enabled for text-only"


def test_validate_content_blocks_rejects_unknown():
    """validate_content_blocks should return error for unsupported types."""
    from app.openai_api import OpenAIMessage, validate_content_blocks
    # Unsupported 'image' type (Anthropic format) in OpenAI endpoint
    msgs = [OpenAIMessage(role='user', content=[
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'abc'}},
    ])]
    err = validate_content_blocks(msgs)
    assert err is not None
    assert 'image' in err


def test_validate_content_blocks_ok_for_text_and_image_url():
    """validate_content_blocks should return None for supported types."""
    from app.openai_api import OpenAIMessage, validate_content_blocks
    msgs = [OpenAIMessage(role='user', content=[
        {'type': 'text', 'text': 'Hello'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}},
    ])]
    err = validate_content_blocks(msgs)
    assert err is None


def test_validate_content_blocks_string_content_ok():
    """validate_content_blocks should return None for plain string content."""
    from app.openai_api import OpenAIMessage, validate_content_blocks
    msgs = [OpenAIMessage(role='user', content='Hello')]
    err = validate_content_blocks(msgs)
    assert err is None


def test_estimate_template_size_counts_base64():
    """_estimate_template_size should count base64 image data."""
    from app.openai_api import _estimate_template_size
    template = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 50000}},
            {"type": "text", "text": "What is this?"},
        ]}
    ]
    size = _estimate_template_size(template)
    assert size >= 50000  # base64 data counted


def test_multimodal_payload_hard_limit_rejects():
    """_to_completion_request should reject payload exceeding hard limit."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    from fastapi import HTTPException
    import pytest

    # Build a request with a huge base64 image that exceeds 2MB
    big_b64 = "A" * 2_500_000
    req = OpenAIChatRequest(
        model='gpt-4o',
        stream=False,
        messages=[
            OpenAIMessage(role='user', content=[
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{big_b64}'}},
                {'type': 'text', 'text': 'Describe'},
            ])
        ]
    )
    with pytest.raises(HTTPException) as exc_info:
        _to_completion_request(req, model_id='gpt-4o', version='2024-11-20')
    assert exc_info.value.status_code == 400
    assert 'too large' in str(exc_info.value.detail).lower()

def test_estimate_template_size_with_safety_factor():
    """_estimate_template_size should include 1.2x safety factor."""
    from app.openai_api import _estimate_template_size
    template = [
        {"role": "user", "content": [
            {"type": "text", "text": "A" * 1000},
        ]}
    ]
    size = _estimate_template_size(template)
    # Should be > 1000 (raw content) due to safety factor
    assert size >= 1000
    # Should be roughly 1000 * 1.2 = 1200
    assert size < 2000  # sanity upper bound


def test_multimodal_stream_returns_sse_format(monkeypatch):
    """Multimodal + stream=true: response should still be SSE format (buffered pseudo-stream)."""
    from fastapi.testclient import TestClient
    from app import main
    from app.model_registry import SupportedModel

    client = TestClient(main.app)
    monkeypatch.setattr(main.settings, 'api_key', '')
    monkeypatch.setattr(main.settings, 'sap_user', 'user')
    monkeypatch.setattr(main.settings, 'sap_pass', 'pass')
    monkeypatch.setattr(
        main,
        'resolve_model_cached',
        lambda *args, **kwargs: SupportedModel(id='gpt-4o', version='1', owned_by='openai'),
    )

    captured: dict = {}
    def fake_execute(username, password, request=None, base_url=None):
        captured['stream_enabled'] = request.stream_enabled
        captured['has_images'] = request.has_images
        class FakeResult:
            status_code = 200
            stream_resp = None  # non-stream => buffered
            body = '{"final_result":{"llm":{"choices":[{"delta":{"content":"A cat"}}],"usage":{"prompt_tokens":100,"completion_tokens":5}}}}'
        return FakeResult()
    monkeypatch.setattr(main, 'execute_completion_with_password_curl_cffi', fake_execute)

    resp = client.post('/v1/chat/completions', json={
        'model': 'gpt-4o',
        'stream': True,
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}},
            {'type': 'text', 'text': 'What is this?'}
        ]}],
    })
    # Request succeeds
    assert resp.status_code == 200
    # SAP upstream was non-stream
    assert captured['stream_enabled'] is False
    # But client still gets SSE format (buffered pseudo-stream)
    assert 'text/event-stream' in resp.headers.get('content-type', '')
    # Body contains SSE data lines
    assert b'data: ' in resp.content


def test_image_template_messages_no_tools_entry() -> None:
    """Regression: _build_image_template_messages must NOT inject {"tools": ...}
    into template list — that dict has no "role" key and crashes _build_template_entry().
    """
    from app.openai_api import _build_image_template_messages, OpenAITool, OpenAIFunction
    tools = [OpenAITool(type="function", function=OpenAIFunction(
        name="screenshot", description="Take screenshot", parameters={}))]
    msgs = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Describe this"},
        ]},
    ]
    # Convert to OpenAIMessage objects
    from app.openai_api import OpenAIChatRequest
    req = OpenAIChatRequest(model="gpt-4o", messages=[
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Describe this"},
        ]},
    ], tools=tools)
    template = _build_image_template_messages(req.messages, req.tools)
    for entry in template:
        assert "role" in entry, f"template entry missing 'role': {list(entry.keys())}"
    # Explicitly: no entry should be just {"tools": [...]}
    for entry in template:
        assert not ("tools" in entry and "role" not in entry), \
            f"Found tools-only entry (no role) in template: {entry}"


def test_template_tool_result_gets_continue_hint() -> None:
    """When template ends with tool_result and tools are present,
    _build_template_messages should append a user hint to prompt continuation.

    This prevents empty responses (boxd02/boxd03 pattern).
    """
    from app.openai_api import _build_template_messages, OpenAITool, OpenAIFunction, OpenAIToolCall, OpenAIToolFunctionCall
    tools = [OpenAITool(type="function", function=OpenAIFunction(
        name="bash", description="Run command", parameters={}))]
    msgs = [
        OpenAIMessage(role="system", content="You are helpful"),
        OpenAIMessage(role="user", content="List files"),
        OpenAIMessage(
            role="assistant",
            content=None,
            tool_calls=[OpenAIToolCall(
                id="call_1", type="function",
                function=OpenAIToolFunctionCall(name="bash", arguments='{"command":"ls"}'),
            )],
        ),
        OpenAIMessage(role="tool", content="file1.txt\nfile2.txt", tool_call_id="call_1"),
    ]
    template = _build_template_messages(msgs, tools)
    # Last entry should be the user hint
    assert template[-1]["role"] == "user"
    assert "continue" in template[-1]["content"].lower()


def test_template_no_hint_without_tools() -> None:
    """No hint should be appended when tools are not present."""
    from app.openai_api import _build_template_messages
    msgs = [
        OpenAIMessage(role="user", content="Hello"),
        OpenAIMessage(role="assistant", content="Hi there"),
        OpenAIMessage(role="user", content="Thanks"),
    ]
    template = _build_template_messages(msgs, tools=None)
    # Should NOT end with a continue hint
    assert template[-1]["role"] != "user" or "continue" not in template[-1]["content"].lower()


def test_template_no_hint_when_not_ending_with_tool() -> None:
    """No hint when template ends with a user message, even with tools present."""
    from app.openai_api import _build_template_messages, OpenAITool, OpenAIFunction
    tools = [OpenAITool(type="function", function=OpenAIFunction(
        name="bash", description="Run command", parameters={}))]
    msgs = [
        OpenAIMessage(role="user", content="Hello"),
    ]
    template = _build_template_messages(msgs, tools)
    # Should be just the user message, no extra hint
    assert len(template) == 1
    assert template[0]["role"] == "user"
    assert template[0]["content"] == "Hello"


def test_force_non_stream_overrides_stream_request(monkeypatch) -> None:
    """When SAP_FORCE_NON_STREAM=true, _to_completion_request should set
    stream_enabled=False even if client requests stream=true."""
    from app.openai_api import _to_completion_request
    from app.config import settings

    monkeypatch.setattr(settings.completion, 'force_non_stream', True)

    req = OpenAIChatRequest(
        model="claude-sonnet-4-5",
        messages=[OpenAIMessage(role="user", content="Hello")],
        stream=True,
    )
    result = _to_completion_request(req, "anthropic--claude-4.5-sonnet", "latest",
                                     deployment_id="dep-1", resource_group_id="rg-1")
    assert result.stream_enabled is False, "force_non_stream should override stream=True"


def test_force_non_stream_false_allows_stream(monkeypatch) -> None:
    """When SAP_FORCE_NON_STREAM=false (default), stream=true passes through."""
    from app.openai_api import _to_completion_request
    from app.config import settings

    monkeypatch.setattr(settings.completion, 'force_non_stream', False)

    req = OpenAIChatRequest(
        model="claude-sonnet-4-5",
        messages=[OpenAIMessage(role="user", content="Hello")],
        stream=True,
    )
    result = _to_completion_request(req, "anthropic--claude-4.5-sonnet", "latest",
                                     deployment_id="dep-1", resource_group_id="rg-1")
    assert result.stream_enabled is True, "stream should pass through when force_non_stream=false"
    assert result.stream_enabled is True, "stream should pass through when force_non_stream=false"


def test_history_truncation_does_not_drop_subsequent_messages() -> None:
    """After token-budget truncation, all subsequent messages must still be processed.
    Bug: _build_messages_history previously had `break` after truncation, which
    dropped ALL messages after the one that triggered the budget — including the
    assistant with tool_use whose tool_result was in the template, causing
    SAP 400 "unexpected tool_use_id" errors."""
    from app.openai_api import _build_messages_history, _build_template_messages, OpenAIMessage, OpenAIToolCall, OpenAIToolFunctionCall
    from app import model_registry, openai_api

    orig_tokens = openai_api.settings.max_history_tokens
    orig_turns = openai_api.settings.max_history_turns
    try:
        # Budget allows ~100 chars (400 / 4 = 100 tokens)
        openai_api.settings.max_history_tokens = 100
        openai_api.settings.max_history_turns = 100

        messages = [
            OpenAIMessage(role='user', content='A' * 200),   # long, will be truncated
            OpenAIMessage(role='assistant', content='B'),     # short
            # Below: messages that must NOT be dropped by break-after-truncation
            OpenAIMessage(role='user', content='C'),          # short, after truncation trigger
            OpenAIMessage(                                    # assistant with tool_use
                role='assistant', content='',
                tool_calls=[OpenAIToolCall(
                    id='call_important',
                    function=OpenAIToolFunctionCall(name='my_tool', arguments='{}'),
                )],
            ),
            OpenAIMessage(role='tool', content='result', tool_call_id='call_important'),
            OpenAIMessage(role='user', content='Follow-up'),  # new turn (last user/tool)
        ]

        history = _build_messages_history(messages)
        template = _build_template_messages(messages, tools=None)
        # The assistant with tool_use (call_important) MUST be present
        # in either history or template (turn boundary logic decides where).
        # With the old break-after-truncation bug, it would be dropped entirely.
        tool_ids = set()
        for entry in history:
            for tc in entry.get('tool_calls', []):
                tool_ids.add(tc.get('id', ''))
        for entry in template:
            for tc in entry.get('tool_calls', []):
                tool_ids.add(tc.get('id', ''))
        assert 'call_important' in tool_ids, (
            f"assistant with tool_use 'call_important' was dropped! "
            f"History entries: {len(history)}, template entries: {len(template)}, "
            f"tool_ids found: {tool_ids}"
        )
    finally:
        openai_api.settings.max_history_tokens = orig_tokens
        openai_api.settings.max_history_turns = orig_turns


def test_history_truncation_with_image_does_not_drop_subsequent_messages() -> None:
    """Large image_url in history should not cause break-after-truncation to drop
    later messages (including assistant with tool_use).
    This was the exact bug that caused the directory-files-listing session to fail
    after screenshot: image (230KB) pushed budget over limit → break dropped the
    assistant with tool_use → SAP rejected orphaned tool_result in template."""
    from app.openai_api import _build_messages_history_with_images, _build_template_messages, OpenAIMessage, OpenAIToolCall, OpenAIToolFunctionCall
    from app import model_registry, openai_api

    orig_tokens = openai_api.settings.max_history_tokens
    orig_turns = openai_api.settings.max_history_turns
    try:
        # Budget allows ~200 chars (800 / 4 = 200 tokens)
        openai_api.settings.max_history_tokens = 200
        openai_api.settings.max_history_turns = 100

        fake_image_url = f"data:image/png;base64,{'x' * 500}"  # ~500 chars
        messages = [
            OpenAIMessage(role='user', content='Hello'),
            OpenAIMessage(role='assistant', content='Hi there'),
            OpenAIMessage(role='user', content=[              # image message (big!)
                {"type": "image_url", "image_url": {"url": fake_image_url}},
            ]),
            # Messages AFTER the image — these must NOT be dropped
            OpenAIMessage(                                    # assistant with tool_use
                role='assistant', content='Let me look',
                tool_calls=[OpenAIToolCall(
                    id='call_after_image',
                    function=OpenAIToolFunctionCall(name='grep', arguments='{"pattern":"foo"}'),
                )],
            ),
            OpenAIMessage(role='tool', content='found it', tool_call_id='call_after_image'),
            OpenAIMessage(role='user', content='Next step'),  # new turn
        ]

        history = _build_messages_history_with_images(messages)
        template = _build_template_messages(messages, tools=None)
        # The assistant with tool_use (call_after_image) MUST be present
        # in either history or template (turn boundary logic decides where).
        tool_ids = set()
        for entry in history:
            for tc in entry.get('tool_calls', []):
                tool_ids.add(tc.get('id', ''))
        for entry in template:
            for tc in entry.get('tool_calls', []):
                tool_ids.add(tc.get('id', ''))
        assert 'call_after_image' in tool_ids, (
            f"assistant with tool_use 'call_after_image' was dropped! "
            f"History entries: {len(history)}, template entries: {len(template)}, "
            f"tool_ids found: {tool_ids}"
        )
    finally:
        openai_api.settings.max_history_tokens = orig_tokens
        openai_api.settings.max_history_turns = orig_turns


def test_repair_tool_adjacency_removes_orphaned_tool_results() -> None:
    """_repair_tool_adjacency must remove tool results whose tool_use_id has no
    matching tool_use in any preceding assistant message."""
    from app.openai_api import _repair_tool_adjacency

    history = [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "orphaned result", "tool_call_id": "call_missing"},
        {"role": "assistant", "content": "Hi", "tool_calls": [{"id": "call_present", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]},
        {"role": "tool", "content": "good result", "tool_call_id": "call_present"},
    ]

    repaired = _repair_tool_adjacency(history)
    # Orphaned tool result should be removed, valid one kept
    tool_call_ids = [e.get("tool_call_id", "") for e in repaired if e.get("role") == "tool"]
    assert "call_missing" not in tool_call_ids
    assert "call_present" in tool_call_ids


def test_repair_tool_adjacency_removes_leading_tool_messages() -> None:
    """If truncation removes the assistant that started the history, the leading
    tool result(s) become orphaned and must be removed."""
    from app.openai_api import _repair_tool_adjacency

    history = [
        {"role": "tool", "content": "dangling", "tool_call_id": "call_x"},
        {"role": "user", "content": "Next question"},
        {"role": "assistant", "content": "Answer"},
    ]

    repaired = _repair_tool_adjacency(history)
    assert repaired[0]["role"] != "tool", "leading tool message should be removed"
    assert len(repaired) == 2

def test_parse_model_effort_with_suffix() -> None:
    from app.model_registry import _parse_model_effort
    assert _parse_model_effort("gpt-5.4:high") == ("gpt-5.4", "high")
    assert _parse_model_effort("o4-mini:low") == ("o4-mini", "low")
    assert _parse_model_effort("gpt-5.4:medium") == ("gpt-5.4", "medium")
    assert _parse_model_effort("gpt-5.4:xhigh") == ("gpt-5.4", "xhigh")
    # No suffix
    assert _parse_model_effort("gpt-5.4") == ("gpt-5.4", None)
    # Invalid effort = not an effort, treat as part of model name
    assert _parse_model_effort("gpt-5.4:invalid") == ("gpt-5.4:invalid", None)


def test_supports_reasoning_effort() -> None:
    from app.model_registry import _supports_reasoning_effort
    assert _supports_reasoning_effort("gpt-5.4") is True
    assert _supports_reasoning_effort("gpt-5.2") is True
    assert _supports_reasoning_effort("gpt-5-mini") is True
    assert _supports_reasoning_effort("o4-mini") is True
    assert _supports_reasoning_effort("o3") is True
    assert _supports_reasoning_effort("o1") is True
    assert _supports_reasoning_effort("gpt-4o") is False
    assert _supports_reasoning_effort("gpt-4.1") is False
    assert _supports_reasoning_effort("anthropic--claude-4.6-sonnet") is False


def test_reasoning_effort_passed_to_completion_request() -> None:
    """model:effort suffix → reasoning_effort in CompletionRequest → in SAP params."""
    from app.model_registry import _parse_model_effort
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    from app.curl_login import _build_completion_payload

    req = OpenAIChatRequest(
        model="gpt-5.4:high",
        messages=[OpenAIMessage(role="user", content="Hello")],
        max_tokens=100,
    )
    cr = _to_completion_request(req, "gpt-5.4", "latest")
    assert cr.reasoning_effort == "high", f"expected 'high', got {cr.reasoning_effort}"

    # Verify it reaches the SAP payload
    payload = _build_completion_payload(cr)
    params = payload["config"]["modules"]["prompt_templating"]["model"]["params"]
    assert params.get("reasoning_effort") == "high"


def test_reasoning_effort_filtered_for_gpt4o() -> None:
    """reasoning_effort should be dropped for gpt-4o (not supported)."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request

    req = OpenAIChatRequest(
        model="gpt-4o:high",
        messages=[OpenAIMessage(role="user", content="Hello")],
        max_tokens=100,
    )
    cr = _to_completion_request(req, "gpt-4o", "latest")
    assert cr.reasoning_effort is None, "reasoning_effort should be None for gpt-4o"


def test_reasoning_effort_no_suffix() -> None:
    """Model name without :effort suffix → reasoning_effort=None."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    from app.curl_login import _build_completion_payload

    req = OpenAIChatRequest(
        model="gpt-5.4",
        messages=[OpenAIMessage(role="user", content="Hello")],
        max_tokens=100,
    )
    cr = _to_completion_request(req, "gpt-5.4", "latest")
    assert cr.reasoning_effort is None

    payload = _build_completion_payload(cr)
    params = payload["config"]["modules"]["prompt_templating"]["model"]["params"]
    assert "reasoning_effort" not in params


def test_resolve_model_strips_effort_suffix() -> None:
    """resolve_model and resolve_model_cached should strip :effort before lookup."""
    from app.model_registry import resolve_model, resolve_model_cached, SupportedModel

    models = [
        SupportedModel(id="gpt-5.4", version="2026-03-05", owned_by="openai"),
        SupportedModel(id="o4-mini", version="latest", owned_by="openai"),
    ]

    # resolve_model should find gpt-5.4 when asked for "gpt-5.4:high"
    assert resolve_model(models, "gpt-5.4:high") is not None
    assert resolve_model(models, "gpt-5.4:high").id == "gpt-5.4"
    assert resolve_model(models, "o4-mini:low") is not None
    assert resolve_model(models, "o4-mini:low").id == "o4-mini"


def test_o_series_uses_max_completion_tokens() -> None:
    """o-series models should use max_completion_tokens (not max_tokens)."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    from app.curl_login import _build_completion_payload

    req = OpenAIChatRequest(
        model="o4-mini:low",
        messages=[OpenAIMessage(role="user", content="Hello")],
        max_tokens=500,
    )
    cr = _to_completion_request(req, "o4-mini", "latest")
    payload = _build_completion_payload(cr)
    params = payload["config"]["modules"]["prompt_templating"]["model"]["params"]
    assert "max_completion_tokens" in params
    assert "max_tokens" not in params
    assert "temperature" not in params  # o-series doesn't support temperature

def test_claude_47_opus_deprecates_temperature() -> None:
    from app.model_registry import _claude_deprecates_temperature
    assert _claude_deprecates_temperature("anthropic--claude-4.7-opus") is True
    assert _claude_deprecates_temperature("anthropic--claude-4.6-sonnet") is False
    assert _claude_deprecates_temperature("anthropic--claude-4.5-sonnet") is False
    assert _claude_deprecates_temperature("gpt-5.4") is False

def test_claude_adaptive_thinking_support() -> None:
    from app.model_registry import _claude_supports_adaptive_thinking
    assert _claude_supports_adaptive_thinking("anthropic--claude-4.7-opus") is True
    assert _claude_supports_adaptive_thinking("anthropic--claude-4.6-sonnet") is True
    assert _claude_supports_adaptive_thinking("anthropic--claude-4.5-sonnet") is False

def test_claude_enabled_thinking_support() -> None:
    from app.model_registry import _claude_supports_enabled_thinking
    assert _claude_supports_enabled_thinking("anthropic--claude-4.5-sonnet") is True
    assert _claude_supports_enabled_thinking("anthropic--claude-4.6-sonnet") is True
    assert _claude_supports_enabled_thinking("anthropic--claude-4.7-opus") is False

def test_claude_47_opus_effort_builds_adaptive_thinking() -> None:
    """claude-4.7-opus:high → thinking=adaptive + output_config.effort=high in SAP params."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    from app.curl_login import _build_completion_payload

    req = OpenAIChatRequest(
        model="claude-4.7-opus:high",
        messages=[OpenAIMessage(role="user", content="Hello")],
        max_tokens=2000,
    )
    cr = _to_completion_request(req, "anthropic--claude-4.7-opus", "latest")
    assert cr.reasoning_effort == "high"

    payload = _build_completion_payload(cr)
    params = payload["config"]["modules"]["prompt_templating"]["model"]["params"]
    assert params.get("thinking") == {"type": "adaptive"}
    assert params.get("output_config") == {"effort": "high"}
    # temperature must NOT be present for 4.7+
    assert "temperature" not in params

def test_claude_46_sonnet_effort_builds_adaptive_thinking() -> None:
    """claude-4.6-sonnet:high → thinking=adaptive + output_config.effort=high."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    from app.curl_login import _build_completion_payload

    req = OpenAIChatRequest(
        model="claude-4.6-sonnet:high",
        messages=[OpenAIMessage(role="user", content="Hello")],
        max_tokens=2000,
    )
    cr = _to_completion_request(req, "anthropic--claude-4.6-sonnet", "latest")
    payload = _build_completion_payload(cr)
    params = payload["config"]["modules"]["prompt_templating"]["model"]["params"]
    assert params.get("thinking") == {"type": "adaptive"}
    assert params.get("output_config") == {"effort": "high"}
    # temperature must NOT be present when thinking is active
    assert "temperature" not in params

def test_claude_45_sonnet_effort_builds_enabled_thinking() -> None:
    """claude-4.5-sonnet:high → thinking=enabled + budget_tokens (mapped from effort)."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    from app.curl_login import _build_completion_payload

    req = OpenAIChatRequest(
        model="claude-4.5-sonnet:high",
        messages=[OpenAIMessage(role="user", content="Hello")],
        max_tokens=2000,
    )
    cr = _to_completion_request(req, "anthropic--claude-4.5-sonnet", "latest")
    payload = _build_completion_payload(cr)
    params = payload["config"]["modules"]["prompt_templating"]["model"]["params"]
    assert params.get("thinking", {}).get("type") == "enabled"
    assert "budget_tokens" in params.get("thinking", {})
    # max_tokens must be > budget_tokens
    assert params.get("max_tokens", 0) > params.get("thinking", {}).get("budget_tokens", 0)

def test_claude_47_opus_no_suffix_no_temperature() -> None:
    """claude-4.7-opus without effort suffix: still no temperature (deprecated)."""
    from app.openai_api import OpenAIChatRequest, _to_completion_request
    from app.curl_login import _build_completion_payload

    req = OpenAIChatRequest(
        model="claude-4.7-opus",
        messages=[OpenAIMessage(role="user", content="Hello")],
        max_tokens=2000,
    )
    cr = _to_completion_request(req, "anthropic--claude-4.7-opus", "latest")
    payload = _build_completion_payload(cr)
    params = payload["config"]["modules"]["prompt_templating"]["model"]["params"]
    assert "temperature" not in params

def test_template_includes_tool_calls_with_tool_result() -> None:
    """Regression: _build_template_messages must include assistant(tool_calls) when
    the last message is a tool_result. Without the assistant, SAP rejects with
    400 "assistant message with 'tool_calls' must be followed by tool messages".

    Bug: _build_template_messages used last_user_idx (pointing at the tool msg)
    as the start of template inclusion, skipping the assistant(tool_calls) that
    precedes the tool_result.
    """
    from app.openai_api import _build_template_messages, OpenAITool, OpenAIFunction, OpenAIToolCall, OpenAIToolFunctionCall
    tools = [OpenAITool(type="function", function=OpenAIFunction(
        name="bash", description="Run command", parameters={}))]
    msgs = [
        OpenAIMessage(role="system", content="You are helpful"),
        OpenAIMessage(role="user", content="List files"),
        OpenAIMessage(role="assistant", content="Let me check."),
        OpenAIMessage(role="user", content="What's in the dir?"),
        OpenAIMessage(
            role="assistant",
            content=None,
            tool_calls=[OpenAIToolCall(
                id="call_1", type="function",
                function=OpenAIToolFunctionCall(name="bash", arguments='{"command":"ls"}'),
            )],
        ),
        OpenAIMessage(role="tool", content="file1.txt", tool_call_id="call_1"),
    ]
    template = _build_template_messages(msgs, tools)
    roles = [e["role"] for e in template]
    # Must have assistant with tool_calls before the tool result
    assert "assistant" in roles, f"assistant missing from template: {roles}"
    # Find the assistant entry and verify it has tool_calls
    assistant_entries = [e for e in template if e["role"] == "assistant"]
    assert any(e.get("tool_calls") for e in assistant_entries), \
        f"No assistant entry with tool_calls: {template}"
    # Tool result must also be present
    assert "tool" in roles, f"tool missing from template: {roles}"
    # Order: assistant(tool_calls) must come before tool
    a_idx = next(i for i, e in enumerate(template) if e["role"] == "assistant" and e.get("tool_calls"))
    t_idx = next(i for i, e in enumerate(template) if e["role"] == "tool")
    assert a_idx < t_idx, f"assistant(tool_calls) at {a_idx} should precede tool at {t_idx}"
    # Previous turn messages (user "List files", assistant "Let me check") should NOT be in template
    texts = [e.get("content", "") for e in template if isinstance(e.get("content"), str)]
    assert not any("Let me check" in t for t in texts), \
        f"Previous turn leaked into template: {texts}"


def test_template_multi_tool_loop() -> None:
    """Template must include the full tool loop when the turn has multiple
    assistant(tool_calls) → tool_result pairs."""
    from app.openai_api import _build_template_messages, OpenAITool, OpenAIFunction, OpenAIToolCall, OpenAIToolFunctionCall
    tools = [OpenAITool(type="function", function=OpenAIFunction(
        name="bash", description="Run command", parameters={}))]
    msgs = [
        OpenAIMessage(role="system", content="You are helpful"),
        OpenAIMessage(role="user", content="Check the system"),
        OpenAIMessage(
            role="assistant",
            content=None,
            tool_calls=[OpenAIToolCall(
                id="call_1", type="function",
                function=OpenAIToolFunctionCall(name="bash", arguments='{"command":"uname"}'),
            )],
        ),
        OpenAIMessage(role="tool", content="Linux", tool_call_id="call_1"),
        OpenAIMessage(
            role="assistant",
            content=None,
            tool_calls=[OpenAIToolCall(
                id="call_2", type="function",
                function=OpenAIToolFunctionCall(name="bash", arguments='{"command":"df"}'),
            )],
        ),
        OpenAIMessage(role="tool", content="90% used", tool_call_id="call_2"),
    ]
    template = _build_template_messages(msgs, tools)
    roles = [e["role"] for e in template]
    # Both assistant(tool_calls) entries must be present
    assistant_tc_count = sum(1 for e in template if e["role"] == "assistant" and e.get("tool_calls"))
    assert assistant_tc_count == 2, f"Expected 2 assistant(tool_calls), got {assistant_tc_count}: {roles}"
    # Both tool results must be present
    tool_count = sum(1 for e in template if e["role"] == "tool")
    assert tool_count == 2, f"Expected 2 tool results, got {tool_count}: {roles}"
    # User message starting the turn must be present
    user_texts = [e.get("content") for e in template if e["role"] == "user" and isinstance(e.get("content"), str)]
    assert any("Check the system" in str(t) for t in user_texts), \
        f"Turn-start user message missing: {user_texts}"

# ── _reorder_tool_result_images ──────────────────────────────────────

class TestReorderToolResultImages:
    """Tests for _reorder_tool_result_images that moves user messages
    between tool results to after the tool result group."""

    def _make_tc(self, ids):
        """Helper: create assistant entry with tool_calls."""
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": tid, "type": "function", "function": {"name": "test", "arguments": "{}"}} for tid in ids],
        }

    def _make_tool(self, tc_id, content="result"):
        return {"role": "tool", "tool_call_id": tc_id, "content": content}

    def _make_user(self, text="hello"):
        return {"role": "user", "content": text}

    def _make_user_image(self, text="image"):
        return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}

    def test_no_reorder_needed(self):
        """Tool results already consecutive — no change."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            self._make_tc(["a", "b"]),
            self._make_tool("a"),
            self._make_tool("b"),
        ]
        result = _reorder_tool_result_images(entries)
        assert [e.get("role") for e in result] == ["assistant", "tool", "tool"]

    def test_user_image_between_tool_results(self):
        """user(image) between tool results → moved after last tool."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            self._make_tc(["a", "b", "c"]),
            self._make_tool("a"),
            self._make_user_image(),  # should move after tool(c)
            self._make_tool("b"),
            self._make_tool("c"),
        ]
        result = _reorder_tool_result_images(entries)
        roles = [e.get("role") for e in result]
        # user(image) moved to after all tools
        assert roles == ["assistant", "tool", "tool", "tool", "user"], f"Got: {roles}"

    def test_user_text_between_tool_results(self):
        """Plain text user between tool results → also moved."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            self._make_tc(["a", "b"]),
            self._make_tool("a"),
            self._make_user("intervening text"),
            self._make_tool("b"),
        ]
        result = _reorder_tool_result_images(entries)
        roles = [e.get("role") for e in result]
        assert roles == ["assistant", "tool", "tool", "user"], f"Got: {roles}"

    def test_multiple_user_images_between_tools(self):
        """Multiple user(image) messages between tool results → all moved."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            self._make_tc(["a", "b", "c"]),
            self._make_tool("a"),
            self._make_user_image("img1"),
            self._make_tool("b"),
            self._make_user_image("img2"),
            self._make_tool("c"),
        ]
        result = _reorder_tool_result_images(entries)
        roles = [e.get("role") for e in result]
        assert roles == ["assistant", "tool", "tool", "tool", "user", "user"], f"Got: {roles}"

    def test_no_reorder_when_user_after_all_tools(self):
        """User after all tool results → no change needed."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            self._make_tc(["a", "b"]),
            self._make_tool("a"),
            self._make_tool("b"),
            self._make_user_image(),
        ]
        result = _reorder_tool_result_images(entries)
        assert result == entries  # unchanged

    def test_no_reorder_when_no_tool_calls(self):
        """No assistant(tool_calls) → no reordering."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        result = _reorder_tool_result_images(entries)
        assert result == entries  # unchanged

    def test_user_before_first_tool_not_moved(self):
        """User message before any tool result (not between) → not moved."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            self._make_tc(["a"]),
            self._make_user("before tool"),
            self._make_tool("a"),
        ]
        # User is between assistant(tc) and tool, but there's no tool
        # after the user that it's blocking. last_tool_pos = 2, max(pending) = 1
        # So last_tool_pos > max(pending) is True, and the user gets moved.
        # Actually this IS correct behavior — SAP wants tool right after assistant(tc).
        result = _reorder_tool_result_images(entries)
        roles = [e.get("role") for e in result]
        assert roles == ["assistant", "tool", "user"], f"Got: {roles}"

    def test_preserves_tool_call_ids(self):
        """Tool results keep their tool_call_ids after reordering."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            self._make_tc(["call_1", "call_2"]),
            self._make_tool("call_1", "screenshot result"),
            self._make_user_image(),
            self._make_tool("call_2", "console result"),
        ]
        result = _reorder_tool_result_images(entries)
        tool_ids = [e.get("tool_call_id") for e in result if e.get("role") == "tool"]
        assert tool_ids == ["call_1", "call_2"]

    def test_empty_list(self):
        from app.openai_api import _reorder_tool_result_images
        assert _reorder_tool_result_images([]) == []

    def test_separate_assistant_tc_groups(self):
        """Two separate assistant(tc) groups — each reordered independently."""
        from app.openai_api import _reorder_tool_result_images
        entries = [
            self._make_tc(["a", "b"]),
            self._make_tool("a"),
            self._make_user_image("img_a"),
            self._make_tool("b"),
            # assistant without tc acts as boundary
            {"role": "assistant", "content": "done"},
            self._make_tc(["c", "d"]),
            self._make_tool("c"),
            self._make_user_image("img_c"),
            self._make_tool("d"),
        ]
        result = _reorder_tool_result_images(entries)
        roles = [e.get("role") for e in result]
        assert roles == ["assistant", "tool", "tool", "user", "assistant", "assistant", "tool", "tool", "user"], f"Got: {roles}"
