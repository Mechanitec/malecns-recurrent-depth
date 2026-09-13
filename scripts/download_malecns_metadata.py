from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    for name in FILES:
        dest = out / name
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"downloading {name} ...")
        urlretrieve(BASE + name, dest)
        print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    print("\nThe full edge table is intentionally not auto-downloaded:")
    print(BASE + "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    print("Official size is about 1.1 GB. Place it in data/ when ready.")


if __name__ == "__main__":
    main()
