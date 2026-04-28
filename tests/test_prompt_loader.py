import pytest

from doc_enrichment.errors import PromptLoadError
from doc_enrichment.prompt_loader import get_template, load_prompt


# ---------------------------------------------------------------------------
# load_prompt
# ---------------------------------------------------------------------------

def test_load_document_prompt_returns_dict():
    data = load_prompt("document")
    assert isinstance(data, dict)
    assert data["version"] == "v1"


def test_load_parent_prompt_returns_dict():
    data = load_prompt("parent")
    assert isinstance(data, dict)
    assert "input_variables" in data
    assert "template" in data


def test_load_contrast_prompt_returns_dict():
    data = load_prompt("contrast")
    assert isinstance(data, dict)
    assert "input_variables" in data
    assert "template" in data


def test_load_unknown_prompt_raises():
    with pytest.raises(PromptLoadError, match="not found"):
        load_prompt("nonexistent_prompt")


def test_load_unknown_version_raises():
    with pytest.raises(PromptLoadError, match="not found"):
        load_prompt("document", version="v99")


# ---------------------------------------------------------------------------
# get_template - single-template prompts
# ---------------------------------------------------------------------------

def test_get_parent_template_is_string():
    template = get_template("parent")
    assert isinstance(template, str)
    assert len(template) > 0


def test_get_contrast_template_is_string():
    template = get_template("contrast")
    assert isinstance(template, str)
    assert len(template) > 0


def test_parent_template_contains_expected_variables():
    template = get_template("parent")
    for var in ["{branch_id}", "{left_node_id}", "{right_node_id}", "{left_payload}", "{right_payload}"]:
        assert var in template, f"Expected variable {var!r} not found in parent template"


def test_contrast_template_contains_expected_variables():
    template = get_template("contrast")
    for var in ["{payload_a}", "{payload_b}"]:
        assert var in template, f"Expected variable {var!r} not found in contrast template"


# ---------------------------------------------------------------------------
# get_template - multi-section document prompt
# ---------------------------------------------------------------------------

def test_get_document_extract_template():
    template = get_template("document", sub_key="extract")
    assert isinstance(template, str)
    assert len(template) > 0


def test_get_document_reduce_template():
    template = get_template("document", sub_key="reduce")
    assert isinstance(template, str)
    assert len(template) > 0


def test_document_extract_template_contains_expected_variables():
    template = get_template("document", sub_key="extract")
    for var in ["{document_id}", "{source}", "{title}", "{document_text}"]:
        assert var in template, f"Expected variable {var!r} not found in document extract template"


def test_document_reduce_template_contains_expected_variables():
    template = get_template("document", sub_key="reduce")
    for var in ["{document_id}", "{source}", "{title}", "{chunk_payloads}"]:
        assert var in template, f"Expected variable {var!r} not found in document reduce template"


def test_document_prompt_missing_sub_key_raises():
    with pytest.raises(PromptLoadError, match="template"):
        # document_v1.yaml has no top-level 'template' key
        get_template("document")


def test_get_template_invalid_sub_key_raises():
    with pytest.raises(PromptLoadError, match="template"):
        get_template("document", sub_key="nonexistent_section")


# ---------------------------------------------------------------------------
# Prompt content smoke tests
# ---------------------------------------------------------------------------

def test_contrast_template_preserves_json_example_braces():
    """Verify that literal {{ }} in the JSON schema example survive YAML loading."""
    template = get_template("contrast")
    assert "{{" in template, "Escaped braces for JSON example should be present in contrast template"


def test_parent_prompt_input_variables_list():
    data = load_prompt("parent")
    expected = {"branch_id", "left_node_id", "right_node_id", "left_payload", "right_payload"}
    assert set(data["input_variables"]) == expected


def test_contrast_prompt_input_variables_list():
    data = load_prompt("contrast")
    assert set(data["input_variables"]) == {"payload_a", "payload_b"}
