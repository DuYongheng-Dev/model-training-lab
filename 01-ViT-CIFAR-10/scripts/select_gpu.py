"""Print the locally assigned GPU UUID after checking its identity and occupancy."""
import json
from pathlib import Path
import subprocess

project = Path(__file__).resolve().parents[1]
selection_path = project / '.local/device.json'
if not selection_path.exists():
    raise SystemExit('No GPU assignment recorded for this machine; obtain an explicit user selection first.')
selection = json.loads(selection_path.read_text())
uuid = selection['gpu_uuid']
if selection['device'] != 'cuda' or not uuid.startswith('GPU-'):
    raise SystemExit('Invalid local GPU assignment')
record = subprocess.check_output([
    'nvidia-smi', '-i', uuid, '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'
], text=True).strip().split(',')
if record[0].strip() != uuid or int(record[1].strip()) < 2048:
    raise SystemExit('Assigned GPU identity or free memory check failed; no alternative GPU selected.')
processes = subprocess.check_output([
    'nvidia-smi', '-i', uuid, '--query-compute-apps=pid', '--format=csv,noheader,nounits'
], text=True).strip()
if processes:
    raise SystemExit('The assigned GPU already has a compute process; inspect its current use before launching.')
print(uuid)
