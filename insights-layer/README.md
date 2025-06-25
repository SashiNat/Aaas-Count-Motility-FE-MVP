# Insights Layer

This module generates clinical insights based on WHO 5th Edition sperm parameters.

## Files
- `count_insights.yaml`: Maps sperm concentration (millions/mL) to insights
- `motility_insights.yaml`: Maps progressive motility % to interpretations
- `morphology_insights.yaml`: Maps % normal forms to comments

## Purpose
These insights are embedded in the test reports for clinician use only. They are not a substitute for medical advice.

## Usage
Use `insights_generator.py` to parse a result and get the matched insight based on the YAML file.

## Disclaimer
These insights are informational aids. Final interpretation must be done by a licensed healthcare provider.
