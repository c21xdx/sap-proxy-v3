from app.curl_login import CompletionRequest, CurlCompletionResult


def test_completion_result_fields() -> None:
    result = CurlCompletionResult(
        final_url='https://example.com/aic/index.html',
        completion_status=200,
        completion_content_type='text/event-stream',
        csrf_token_present=True,
        response_preview='data: {"ok": true}',
    )
    assert result.completion_status == 200
    assert result.csrf_token_present is True
    assert result.response_preview.startswith('data:')


def test_completion_request_fields() -> None:
    request = CompletionRequest(
        prompt_text='Reply with OK',
        model_name='openai--gpt-5.2',
        model_version='1',
        max_tokens=16384,
        temperature=0.2,
        deployment_id='dep-123',
        workspace='aicore',
        resource_group_id='doc-grounding',
        stream_enabled=False,
    )
    assert request.model_name == 'openai--gpt-5.2'
    assert request.max_tokens == 16384
    assert request.stream_enabled is False
