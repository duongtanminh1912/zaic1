"""Verify independent expected IDs, technical statuses and CLI/notebook parity."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'ops/src'))
from lifemate.runtime.configuration import strict_json
from lifemate.runtime.container_io import check_timed_output


def check(result, expected):
    result = Path(result)
    ids = [strict_json(line)['id'] for line in Path(expected).read_text().splitlines()]
    records = []
    for filename in ('submission.jsonl', 'jupyter_submission.jsonl'):
        path = result / filename
        check_timed_output(path, ids)
        rows = [strict_json(line) for line in path.read_text().splitlines()]
        diagnostics = [strict_json(line) for line in
                       (result / (filename + '.diagnostics.jsonl')).read_text().splitlines()]
        if [d['id'] for d in diagnostics] != ids:
            raise ValueError('Diagnostics coverage mismatch: ' + filename)
        failed = [d['id'] for d,row in zip(diagnostics,rows)
                  if d['failure'] or d.get('error_type') or d.get('baseline')!='52'
                  or d.get('status') not in ('finished','asked_missing_info','call_budget_exhausted')
                  or row['time']>30000]
        if failed:
            raise ValueError('Technical failure/timeout; preserve logs: ' + repr(failed))
        records.append([{k: row[k] for k in ('id', 'tool_calls', 'state_hash')} for row in rows])
    if records[0] != records[1]:
        raise ValueError('Predict/notebook differ; preserve both outputs')
    return {'technical_smoke_passed': True, 'matching_rows': len(ids),
            'quality_score_verified': False, 'gpu_timing_scope': 'local GPU only'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--expected', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(check(a.result, a.expected)))
