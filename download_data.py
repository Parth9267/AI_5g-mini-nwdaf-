"""Download the original authors' CSVs at a fixed commit (no synthetic data)."""
import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
COMMIT = "d6fae0dbd7cf70343201f9351fe60498b628bb86"
REPO = "HenokDanielbfg/5g-testbed-conference"
FOLDER = "core dataset/03 Feb 2025 - 18 Feb 2025"


def download():
    destination = ROOT / "dataset"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": f"https://github.com/{REPO}", "commit": COMMIT,
                "folder": FOLDER, "files": {}}
    for name in ("df_location.csv", "df_reg.csv"):
        path = f"{FOLDER}/{name}"
        url = f"https://api.github.com/repos/{REPO}/contents/{quote(path)}?ref={COMMIT}"
        request = Request(url, headers={"User-Agent": "Mini-NWDAF-educational-project"})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        data = base64.b64decode(payload["content"])
        (destination / name).write_bytes(data)
        manifest["files"][name] = {
            "source": f"https://github.com/{REPO}/blob/{COMMIT}/{quote(path)}",
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        print(f"Downloaded {name}: {len(data):,} bytes")
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    download()
