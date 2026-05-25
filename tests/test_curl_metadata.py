from app.curl_login import CurlMetadataResult


def test_metadata_result_fields() -> None:
    result = CurlMetadataResult(
        final_url='https://example.com/aic/index.html',
        metadata_status=200,
        metadata_content_type='application/json',
        model_count=44,
    )
    assert result.metadata_status == 200
    assert result.model_count == 44
