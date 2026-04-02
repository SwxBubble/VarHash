import argparse
import base64
import hashlib
import json
import mmap
import re
from pathlib import Path


def version_key(name):
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if part == "":
            continue
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def strip_archive_suffix(path):
    name = path.name
    for suffix in (".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".tbz2", ".txz", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def iter_tar_files(tar_root):
    for project_dir in sorted(Path(tar_root).iterdir()):
        if not project_dir.is_dir():
            continue
        tar_files = []
        for path in sorted(project_dir.iterdir()):
            if not path.is_file():
                continue
            if any(path.name.endswith(suffix) for suffix in
                   (".tar", ".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".tbz2", ".txz")):
                tar_files.append(path)
        tar_files.sort(key=lambda item: version_key(strip_archive_suffix(item)))
        for version_order, tar_path in enumerate(tar_files):
            yield {
                "project": project_dir.name,
                "version": strip_archive_suffix(tar_path),
                "version_order": version_order,
                "tar_path": tar_path,
            }


def iter_fastcdc_chunks(file_path, min_size, avg_size, max_size, fat):
    try:
        from fastcdc import fastcdc
    except ImportError as exc:
        raise RuntimeError(
            "The 'fastcdc' package is required. Install it with 'pip install fastcdc'."
        ) from exc

    with open(file_path, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for chunk in fastcdc(mapped, min_size=min_size, avg_size=avg_size,
                                 max_size=max_size, fat=fat):
                offset = getattr(chunk, "offset", None)
                length = getattr(chunk, "length", None)
                data = getattr(chunk, "data", None)
                if data is None:
                    if offset is None or length is None:
                        raise RuntimeError("Unsupported fastcdc chunk object shape.")
                    data = mapped[offset: offset + length]
                if offset is None:
                    offset = 0
                if length is None:
                    length = len(data)
                yield offset, length, bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar-root", required=True,
                        help="Root directory like test_data/tar_versions")
    parser.add_argument("--output", required=True,
                        help="Output chunks.jsonl path")
    parser.add_argument("--min-size", type=int, default=4096)
    parser.add_argument("--avg-size", type=int, default=8192)
    parser.add_argument("--max-size", type=int, default=16384)
    parser.add_argument("--fat", action="store_true",
                        help="Enable FastCDC normalization if supported by the package")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_id = 0
    with output_path.open("w", encoding="utf-8") as out:
        for item in iter_tar_files(args.tar_root):
            tar_path = item["tar_path"]
            rel_tar_path = tar_path.relative_to(Path(args.tar_root))
            for offset, length, payload in iter_fastcdc_chunks(
                tar_path, args.min_size, args.avg_size, args.max_size, args.fat
            ):
                sha1_hex = hashlib.sha1(payload).hexdigest()
                record = {
                    "chunk_id": chunk_id,
                    "sha1": sha1_hex,
                    "project": item["project"],
                    "version": item["version"],
                    "version_order": item["version_order"],
                    "tar_path": str(rel_tar_path).replace("\\", "/"),
                    "chunk_offset": offset,
                    "chunk_length": length,
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_id += 1


if __name__ == "__main__":
    main()
