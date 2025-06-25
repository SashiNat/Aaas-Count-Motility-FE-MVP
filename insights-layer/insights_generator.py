import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_insight(result_value, yaml_data, key='min', end_key='max'):
    for rule in yaml_data['thresholds']:
        if rule[key] <= result_value < rule[end_key]:
            return rule['insight']
    return "No insight available for this range."

# Example usage
# count_yaml = load_yaml("count_insights.yaml")
# print(get_insight(12, count_yaml))
