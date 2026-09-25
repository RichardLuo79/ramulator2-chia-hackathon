from ramulator_chia.public_export import redact, redact_text


def test_private_fields_and_embedded_credentials():
    x = redact({'access_token': 'do-not-export', 'nested': [{'cycle': 17,
        'message': 'Bearer abcdefghijklmnop /home/researcher/private/file'}]})
    assert x['access_token'] == '[redacted-private-field]'
    assert x['nested'][0]['cycle'] == 17
    assert 'abcdefghijklmnop' not in str(x)
    assert '/home/researcher' not in str(x)


def test_scientific_text_retained():
    text = 'read latency 42; DDR5_4800AN; tools: train, replay; tool_use_id=abc-123'
    assert redact_text(text) == text
