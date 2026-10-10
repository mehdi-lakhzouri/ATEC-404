# Remediation (Task AI-01.D.2)

Read-only investigations over curated-dataset-v1.1 / yolo-rock-v1.1. Outputs: `experiments/remediation/`.

```powershell
cd backend
uv run --no-sync python -m analysis.remediation.ai_review --source <taskd_ai_review_36_decisions.json>
uv run --no-sync python -m analysis.remediation.scene_investigation
uv run --no-sync python -m analysis.remediation.threshold_study --workers 8
uv run --no-sync python -m analysis.remediation.masking_audit --workers 8
uv run --no-sync python -m analysis.evaluation.build_ignore --workers 8      # PROPOSED sub-threshold ignore regions
uv run --no-sync python -m unittest discover -s tests -t . -v
```
