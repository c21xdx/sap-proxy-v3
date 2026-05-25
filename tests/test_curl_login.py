from app.curl_login import (
    CompletionRequest,
    _build_completion_payload,
    _build_completion_url,
    _build_metadata_url,
    _build_model_params,
    _extract_accounts_authorize,
    _extract_launchpad_bootstrap,
    _launchpad_cookie_domain,
)


def test_extract_launchpad_bootstrap() -> None:
    html = '<script>document.cookie="signature=abc123;path=/;";location="https://example.com/oauth/authorize?x=1"</script>'
    sig, loc = _extract_launchpad_bootstrap(html)
    assert sig == 'abc123'
    assert loc == 'https://example.com/oauth/authorize?x=1'


def test_extract_accounts_authorize_from_href() -> None:
    html = '<a href="https://academy.accounts.ondemand.com/oauth2/authorize?foo=1&amp;bar=2">login</a>'
    url = _extract_accounts_authorize(html)
    assert url == 'https://academy.accounts.ondemand.com/oauth2/authorize?foo=1&bar=2'


def test_launchpad_cookie_domain_from_base_url() -> None:
    assert _launchpad_cookie_domain('https://example.sap.com') == 'example.sap.com'


def test_build_metadata_url() -> None:
    url = _build_metadata_url('https://example.sap.com', 'aicore', 'doc-grounding')
    assert url == 'https://example.sap.com/aic/llm/api/v1/metadataV2?workspace=aicore&resourceGroupId=doc-grounding'


def test_build_completion_url() -> None:
    url = _build_completion_url('https://example.sap.com', 'aicore', 'doc-grounding')
    assert url == 'https://example.sap.com/aic/llm/api/v1/completionV2?workspace=aicore&resourceGroupId=doc-grounding'


def test_build_completion_payload() -> None:
    payload = _build_completion_payload(
        CompletionRequest(
            prompt_text='Write a Python function.',
            model_name='anthropic--claude-4.5-sonnet',
            model_version='1',
            max_tokens=8192,
            temperature=0.2,
            deployment_id='dep-1',
            workspace='aicore',
            resource_group_id='doc-grounding',
            stream_enabled=True,
        )
    )
    # Text mode (no images): model at prompt_templating level (parallel to prompt)
    assert payload['config']['modules']['prompt_templating']['model']['name'] == 'anthropic--claude-4.5-sonnet'
    assert payload['config']['modules']['prompt_templating']['model']['params']['max_tokens'] == 8192
    assert payload['config']['stream']['enabled'] is True


def test_session_cache_reuses_cached_session(monkeypatch) -> None:
    from app import curl_login

    curl_login.session_cache.clear()
    created = []

    class FakeSession:
        pass

    def fake_create(username, password, base_url=None, login_entry_url=None):
        created.append((username, password, base_url, login_entry_url))
        return FakeSession(), 'https://example.com/aic/index.html'

    monkeypatch.setattr(curl_login, 'create_logged_in_session_curl_cffi', fake_create)
    monkeypatch.setattr(curl_login, 'is_session_usable', lambda session, base_url=None: True)
    monkeypatch.setattr(curl_login.settings, 'session_ttl_seconds', 9999)

    s1, u1 = curl_login.get_cached_session_curl_cffi('u', 'p')
    s2, u2 = curl_login.get_cached_session_curl_cffi('u', 'p')

    assert s1 is s2
    assert u1 == u2
    assert len(created) == 1


def test_session_cache_force_refresh(monkeypatch) -> None:
    from app import curl_login

    curl_login.session_cache.clear()
    created = []

    class FakeSession:
        def __init__(self, name: str):
            self.name = name

    def fake_create(username, password, base_url=None, login_entry_url=None):
        name = f'session-{len(created) + 1}'
        created.append(name)
        return FakeSession(name), 'https://example.com/aic/index.html'

    monkeypatch.setattr(curl_login, 'create_logged_in_session_curl_cffi', fake_create)
    monkeypatch.setattr(curl_login.settings, 'session_ttl_seconds', 9999)

    s1, _ = curl_login.get_cached_session_curl_cffi('u', 'p')
    s2, _ = curl_login.get_cached_session_curl_cffi('u', 'p', force_refresh=True)

    assert s1 is not s2
    assert len(created) == 2


def test_session_cache_is_keyed_by_username_and_base_url(monkeypatch) -> None:
    from app import curl_login

    curl_login.session_cache.clear()
    created = []

    class FakeSession:
        def __init__(self, name: str):
            self.name = name

    def fake_create(username, password, base_url=None, login_entry_url=None):
        name = f'{username}@{base_url or "default"}'
        created.append(name)
        return FakeSession(name), f'https://{name}/aic/index.html'

    monkeypatch.setattr(curl_login, 'create_logged_in_session_curl_cffi', fake_create)
    monkeypatch.setattr(curl_login, 'is_session_usable', lambda session, base_url=None: True)
    monkeypatch.setattr(curl_login.settings, 'session_ttl_seconds', 9999)

    s1, _ = curl_login.get_cached_session_curl_cffi('u1', 'p', base_url='https://a.example.com')
    s2, _ = curl_login.get_cached_session_curl_cffi('u1', 'p', base_url='https://a.example.com')
    s3, _ = curl_login.get_cached_session_curl_cffi('u2', 'p', base_url='https://a.example.com')
    s4, _ = curl_login.get_cached_session_curl_cffi('u1', 'p', base_url='https://b.example.com')

    assert s1 is s2
    assert s1 is not s3
    assert s1 is not s4
    assert len(created) == 3


def test_session_cache_snapshot_and_clear_key(monkeypatch) -> None:
    from app import curl_login

    curl_login.session_cache.clear()

    class FakeSession:
        pass

    def fake_create(username, password, base_url=None, login_entry_url=None):
        return FakeSession(), 'https://example.com/aic/index.html'

    monkeypatch.setattr(curl_login, 'create_logged_in_session_curl_cffi', fake_create)
    monkeypatch.setattr(curl_login.settings, 'session_ttl_seconds', 9999)

    curl_login.get_cached_session_curl_cffi('u1', 'p', base_url='https://a.example.com')
    curl_login.get_cached_session_curl_cffi('u2', 'p', base_url='https://b.example.com')

    snapshot = curl_login.session_cache.snapshot()
    assert len(snapshot) == 2
    assert {entry['username'] for entry in snapshot} == {'u1', 'u2'}

    curl_login.session_cache.clear_key('u1', 'https://a.example.com')
    snapshot = curl_login.session_cache.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]['username'] == 'u2'


def test_is_session_usable_success() -> None:
    from app.curl_login import is_session_usable

    class FakeResponse:
        status_code = 200

    class FakeSession:
        def get(self, url):
            assert url.endswith('/aic/api/v1/user')
            return FakeResponse()

    assert is_session_usable(FakeSession(), 'https://example.sap.com') is True


def test_is_session_usable_failure() -> None:
    from app.curl_login import is_session_usable

    class FakeResponse:
        status_code = 401

    class FakeSession:
        def get(self, url):
            return FakeResponse()

    assert is_session_usable(FakeSession(), 'https://example.sap.com') is False


def test_cached_session_validation_refreshes_bad_session(monkeypatch) -> None:
    from app import curl_login

    curl_login.session_cache.clear()
    created = []

    class FakeSession:
        def __init__(self, name: str):
            self.name = name

    def fake_create(username, password, base_url=None, login_entry_url=None):
        name = f'session-{len(created) + 1}'
        created.append(name)
        return FakeSession(name), 'https://example.com/aic/index.html'

    monkeypatch.setattr(curl_login, 'create_logged_in_session_curl_cffi', fake_create)
    monkeypatch.setattr(curl_login, 'is_session_usable', lambda session, base_url=None: getattr(session, 'name', '') == 'session-2')
    monkeypatch.setattr(curl_login.settings, 'session_ttl_seconds', 9999)

    s1, _ = curl_login.get_cached_session_curl_cffi('u', 'p', validate=False)
    s2, _ = curl_login.get_cached_session_curl_cffi('u', 'p', validate=True)

    assert s1 is not s2
    assert len(created) == 2


def test_get_cached_session_wraps_login_error(monkeypatch) -> None:
    from app import curl_login

    curl_login.session_cache.clear()

    def fake_create(username, password, base_url=None, login_entry_url=None):
        raise curl_login.SAPLoginError('login failed')

    monkeypatch.setattr(curl_login, 'create_logged_in_session_curl_cffi', fake_create)

    try:
        curl_login.get_cached_session_curl_cffi('u', 'p', force_refresh=True)
        assert False, 'expected SAPSessionError'
    except curl_login.SAPSessionError as exc:
        assert 'failed to obtain usable SAP session' in str(exc)


def test_execute_completion_raises_completion_error_on_non_200(monkeypatch) -> None:
    from app import curl_login

    class FakeResponse:
        status_code = 500
        text = 'boom'
        headers = {'content-type': 'text/plain'}

        def close(self):
            pass

    class FakeSession:
        def head(self, url, headers=None):
            class HeadResp:
                status_code = 200
                headers = {'x-csrf-token': 't'}
            return HeadResp()

        def post(self, url, json=None, headers=None, stream=False, timeout=None):
            return FakeResponse()

    monkeypatch.setattr(curl_login, 'get_cached_session_curl_cffi', lambda *args, **kwargs: (FakeSession(), 'https://example.com/aic/index.html'))

    try:
        curl_login.execute_completion_with_password_curl_cffi('u', 'p')
        assert False, 'expected SAPCompletionError'
    except curl_login.SAPCompletionError as exc:
        assert 'status 500' in str(exc)


def test_check_model_access_returns_denied_on_completion_error(monkeypatch) -> None:
    from app import curl_login

    def fake_execute(username, password, request=None, base_url=None):
        raise curl_login.SAPCompletionError('completion request failed with status 403')

    monkeypatch.setattr(curl_login, 'execute_completion_with_password_curl_cffi', fake_execute)
    result = curl_login.check_model_access_with_password_curl_cffi('u', 'p', 'm1', '1', force_refresh=True)
    assert result.allowed is False
    assert result.status_code == 403


def test_build_model_params_for_gpt5() -> None:
    params = _build_model_params('gpt-5.4', 64, 0.2)
    assert params == {'max_completion_tokens': 64}


def test_build_model_params_for_claude() -> None:
    params = _build_model_params('anthropic--claude-4.6-opus', 64, 0.2)
    assert params == {'max_tokens': 64, 'temperature': 0.2}


def test_build_completion_payload_multimodal_mode() -> None:
    """When has_images=True, model should be at prompt_templating level,
    and messages_history/placeholder_values should be absent."""
    payload = _build_completion_payload(
        CompletionRequest(
            prompt_text='Describe the image',
            model_name='gpt-4o',
            model_version='latest',
            max_tokens=128,
            temperature=0.2,
            deployment_id='dep-1',
            workspace='aicore',
            resource_group_id='rg-1',
            stream_enabled=True,
            has_images=True,
            template_messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                    {"type": "text", "text": "Describe this"},
                ]},
            ],
        )
    )
    # Model should be at prompt_templating level (parallel to prompt)
    pt = payload['config']['modules']['prompt_templating']
    assert 'model' in pt, "model should be at prompt_templating level in multimodal mode"
    assert pt['model']['name'] == 'gpt-4o'
    # Prompt should NOT have model inside it
    assert 'model' not in pt['prompt'], "model should NOT be inside prompt in multimodal mode"
    # messages_history and placeholder_values should be absent
    assert 'messages_history' not in payload
    assert 'placeholder_values' not in payload['config']
    # stream module should be absent (not allowed in multimodal schema)
    assert 'stream' not in payload['config']['modules']
    # Image URL block should be preserved in template
    template = pt['prompt']['template']
    user_content = template[0]['content']
    img_blocks = [b for b in user_content if b.get('type') == 'image_url']
    assert len(img_blocks) == 1


def test_build_completion_payload_hybrid_mode() -> None:
    """When has_images=True AND messages_history is present (hybrid mode),
    the payload should include both template (images) and messages_history (tools)."""
    payload = _build_completion_payload(
        CompletionRequest(
            prompt_text='Describe the image',
            model_name='anthropic--claude-4.5-sonnet',
            model_version='1',
            max_tokens=128,
            temperature=0.2,
            deployment_id='dep-1',
            workspace='aicore',
            resource_group_id='rg-1',
            stream_enabled=True,
            has_images=True,
            template_messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                    {"type": "text", "text": "Describe this"},
                ]},
            ],
            messages_history=[
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_image", "arguments": "{\"path\":\"/tmp/x.png\"}"}}]},
                {"role": "tool", "content": "Image data", "tool_call_id": "call_1"},
            ],
        )
    )
    pt = payload['config']['modules']['prompt_templating']
    assert 'model' in pt
    # Hybrid mode: messages_history should be PRESENT
    assert 'messages_history' in payload
    assert len(payload['messages_history']) == 2
    # placeholder_values should be present
    assert 'placeholder_values' in payload
    # Stream is present but disabled (multimodal downgrade)
    assert payload['config']['stream']['enabled'] is False
    # Template still has image_url
    template = pt['prompt']['template']
    user_content = template[0]['content']
    img_blocks = [b for b in user_content if b.get('type') == 'image_url']
    assert len(img_blocks) == 1


def test_build_completion_payload_text_mode_unchanged() -> None:
    """When has_images=False (default), text mode format should be unchanged."""
    payload = _build_completion_payload(
        CompletionRequest(
            prompt_text='Hello',
            model_name='anthropic--claude-4.5-sonnet',
            model_version='1',
            max_tokens=4096,
            temperature=0.2,
            deployment_id='dep-1',
            workspace='aicore',
            resource_group_id='rg-1',
            stream_enabled=True,
        )
    )
    # Text mode: model at prompt_templating level (parallel to prompt), NOT inside prompt
    pt = payload['config']['modules']['prompt_templating']
    assert 'model' in pt, "model should be at prompt_templating level"
    assert 'model' not in pt['prompt'], "model should NOT be inside prompt in text mode either"
    assert pt['model']['name'] == 'anthropic--claude-4.5-sonnet'
    # messages_history and stream should be present
    assert 'messages_history' in payload
    assert 'stream' in payload['config']
    assert 'placeholder_values' in payload
