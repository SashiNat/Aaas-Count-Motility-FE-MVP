import yaml
from insights_layer.insights_generator import get_insight, load_yaml

# Load the YAML once for testing
count_yaml = {
    "thresholds": [
        {"min": 0, "max": 15, "insight": "Below WHO reference (15 million/mL). May reduce chances of natural conception. Clinical correlation advised."},
        {"min": 15, "max": 200, "insight": "Within normal WHO range. Fertility depends on motility and morphology as well."},
        {"min": 200, "max": 999, "insight": "Above typical reference range. May be associated with hyperviscosity or other conditions."}
    ]
}

def test_count_low():
    assert get_insight(12, count_yaml) == "Below WHO reference (15 million/mL). May reduce chances of natural conception. Clinical correlation advised."

def test_count_normal():
    assert get_insight(30, count_yaml) == "Within normal WHO range. Fertility depends on motility and morphology as well."

def test_count_high():
    assert get_insight(210, count_yaml) == "Above typical reference range. May be associated with hyperviscosity or other conditions."

print("All insight tests passed ✅")
