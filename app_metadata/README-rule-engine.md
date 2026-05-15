# Rule Engine Usage

This folder contains machine-readable policy and recommendation files plus a local
Python rule engine.

## Files

- compatibility-policy.json
- source-monitoring.json
- dependency-decision-matrix.json
- rule_engine.py
- sample-profile-macos.json

## Run

```bash
python app_metadata/rule_engine.py --profile app_metadata/sample-profile-macos.json
```

Optional output file:

```bash
python app_metadata/rule_engine.py \
  --profile app_metadata/sample-profile-macos.json \
  --output app_metadata/sample-evaluation-output.json
```

## Profile schema (minimum useful fields)

- os: string (example: macos, linux)
- arch: string (example: arm64, x86_64)
- workflow: string or list of strings
- selectedDeps: list of dependency ids/names
- modelVariant: string (example: distilled, base)
- freeDiskTb: number
- licenseAcceptanceRequired: bool
- licenseAccepted: bool
- userPreference.autoRemoveFailedComponents: bool

## Output highlights

- blocked_rules: hard blockers requiring user action
- warnings: non-blocking cautions
- auto_fixes: key-value settings inferred from auto-remediation rules
- dependency_plan: essential/optional/experimental and profile-specific blocked items
- source_plan: ranked sources with effective scores and low-confidence demotions
