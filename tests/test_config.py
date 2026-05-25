from app.config import CompletionModelSettings, CompletionSettings, Settings


def test_settings_login_entry_url_default() -> None:
    settings = Settings(
        sap_base_url='https://example.sap.com',
        sap_login_entry_path='/aic/index.html',
    )
    assert settings.sap_login_entry_url == 'https://example.sap.com/aic/index.html'


def test_completion_model_defaults_are_coding_oriented() -> None:
    model = CompletionModelSettings()
    assert model.max_tokens >= 8192
    assert model.temperature == 0.2


def test_completion_settings_can_be_nested() -> None:
    completion = CompletionSettings(
        workspace='aicore',
        resource_group_id='doc-grounding',
        deployment_id='dep-1',
        stream_enabled=True,
    )
    settings = Settings(completion=completion)
    assert settings.completion.deployment_id == 'dep-1'


def test_session_ttl_setting() -> None:
    settings = Settings(session_ttl_seconds=1234)
    assert settings.session_ttl_seconds == 1234


def test_allowed_models_can_be_configured() -> None:
    settings = Settings(allowed_models=['anthropic--claude-4.6-opus', 'gpt-5.4'])
    assert settings.allowed_models == ['anthropic--claude-4.6-opus', 'gpt-5.4']
