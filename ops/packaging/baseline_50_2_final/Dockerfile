# Build a documentation/training overlay on the exact previously tested 50.2 image.
ARG BASE_IMAGE=zaic-team:rtx4060-test1-fix2
FROM ${BASE_IMAGE}
COPY training_code /code/training_code
COPY README.md /code/README.md
COPY audit_base.py BASE_IMAGE_FILES.json /code/submission_audit/
RUN python3 -B /code/submission_audit/audit_base.py \
    && python3 -B /code/training_code/baseline_50_9_training_repro/audit_public.py \
    && python3 -B /code/training_code/baseline_50_2_training_repro/audit_public.py \
    && python3 -c 'import json; from pathlib import Path; d=json.loads(Path("/code/training_code/DATA_ACCESS.json").read_text()); assert set(d)=={"checkpoints", "shared_training"}; assert all(v["url"].startswith("https://") for v in d.values())'
# Inherit CMD, environment, inference code, dependencies and all model/runtime files.
