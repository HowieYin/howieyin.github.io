"""Download the exact public experiment inputs and check every file hash."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
args.data_dir.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


for manifest_name, subdir in [("model-files.json", "model"), ("dataset-files.json", "")]:
    destination = args.data_dir / subdir
    destination.mkdir(exist_ok=True)
    for entry in json.loads((root / manifest_name).read_text())["files"]:
        path = destination / entry["file"]
        if path.exists() and digest(path) == entry["sha256"]:
            print(f"Verified existing {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "PTQ-Reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024*1024)
        if digest(temporary) != entry["sha256"]:
            raise ValueError(f"Input hash differs from frozen experiment: {path}")
        temporary.replace(path)
        print(f"Downloaded and verified {path}")
for name in ["model-meta.json", "dataset-meta.json"]:
    destination = args.data_dir / name
    if destination.resolve() != (root / name).resolve():
        shutil.copy2(root / name, destination)
print("Inputs verified. Run experiment.py with the same --data-dir.")
