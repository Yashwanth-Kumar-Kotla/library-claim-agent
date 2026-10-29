from google.genai import types

from libraryclaim.live.usage import usage_entry


def test_usage_priced_per_modality():
    u = types.UsageMetadata(
        prompt_token_count=1_100_000, response_token_count=100_000, thoughts_token_count=0,
        prompt_tokens_details=[types.ModalityTokenCount(modality="AUDIO", token_count=100_000),
                               types.ModalityTokenCount(modality="IMAGE", token_count=1_000_000)],
        response_tokens_details=[types.ModalityTokenCount(modality="AUDIO", token_count=100_000)])
    e = usage_entry(u)
    assert e["prompt"] == {"AUDIO": 100_000, "IMAGE": 1_000_000} and e["response"] == {"AUDIO": 100_000}
    assert e["usd"] == round(0.1 * 3.00 + 1.0 * 1.00 + 0.1 * 12.00, 6)


def test_usage_without_details_falls_back_sensibly():
    e = usage_entry(types.UsageMetadata(prompt_token_count=1000, response_token_count=500, thoughts_token_count=200))
    assert e["prompt"] == {"TEXT": 1000} and e["response"] == {"AUDIO": 500} and e["thoughts"] == 200
    assert e["usd"] == round((1000 * 0.75 + 500 * 12 + 200 * 4.5) / 1e6, 6)
