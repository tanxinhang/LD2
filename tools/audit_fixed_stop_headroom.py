"""Fixed-stop family comparison. Not an adaptive or multi-UAV oracle."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.audit_unified_receiver_transport import audit


def run():
    result=audit(samples=100000,stop_headroom=True)
    methods={r['method']:r for r in result['rows']}
    for row in result['fixed_stop_headroom']:
        metric=methods[f"fixed_stop_{row['cut']}"]
        row.update(pd=metric['pd'],pfa=metric['pfa'])
    return dict(rows=result['fixed_stop_headroom'],
        scope='one remote sender; shared samples and fixed link; no held-out schedule selection')


if __name__=='__main__': print(json.dumps(run(),indent=2))
