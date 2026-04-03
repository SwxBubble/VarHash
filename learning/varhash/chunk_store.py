import json
from collections import OrderedDict, defaultdict
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def chunk_sort_key(record):
    return (
        record["project"],
        int(record.get("version_order", 0)),
        int(record.get("chunk_offset", 0)),
        record["sha1"],
    )


def load_grouped_chunk_records(chunks_path, allowed_projects=None,
                               min_version_order=None, max_version_order=None):
    grouped = defaultdict(list)
    by_sha1 = {}
    for record in load_jsonl(chunks_path):
        project = record["project"]
        version_order = int(record.get("version_order", 0))
        if allowed_projects and project not in allowed_projects:
            continue
        if min_version_order is not None and version_order < min_version_order:
            continue
        if max_version_order is not None and version_order > max_version_order:
            continue
        grouped[project].append(record)
        by_sha1.setdefault(record["sha1"], record)
    for project in grouped:
        grouped[project].sort(key=chunk_sort_key)
    return grouped, by_sha1


class TarChunkReader:
    def __init__(self, tar_root, max_open_files=8):
        self.tar_root = Path(tar_root)
        self.max_open_files = max_open_files
        self._handles = OrderedDict()

    def _get_handle(self, relative_path):
        relative_path = str(relative_path).replace("\\", "/")
        handle = self._handles.pop(relative_path, None)
        if handle is None:
            absolute_path = self.tar_root / relative_path
            handle = open(absolute_path, "rb")
        self._handles[relative_path] = handle
        while len(self._handles) > self.max_open_files:
            _, stale_handle = self._handles.popitem(last=False)
            stale_handle.close()
        return handle

    def read(self, relative_path, offset, length):
        handle = self._get_handle(relative_path)
        handle.seek(int(offset))
        payload = handle.read(int(length))
        if len(payload) != int(length):
            raise RuntimeError(
                f"Short read for {relative_path} at offset {offset} "
                f"(expected {length}, got {len(payload)})"
            )
        return payload

    def read_record(self, record):
        return self.read(record["tar_path"], record["chunk_offset"], record["chunk_length"])

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


class ChunkStore:
    def __init__(self, chunks_path, tar_root, max_open_files=8, max_cached_chunks=4096):
        _, self.records_by_sha1 = load_grouped_chunk_records(chunks_path)
        self.reader = TarChunkReader(tar_root, max_open_files=max_open_files)
        self.max_cached_chunks = max_cached_chunks
        self._chunk_cache = OrderedDict()

    def get_record(self, sha1_hex):
        return self.records_by_sha1[sha1_hex]

    def get(self, sha1_hex):
        payload = self._chunk_cache.pop(sha1_hex, None)
        if payload is None:
            payload = self.reader.read_record(self.records_by_sha1[sha1_hex])
        self._chunk_cache[sha1_hex] = payload
        while len(self._chunk_cache) > self.max_cached_chunks:
            self._chunk_cache.popitem(last=False)
        return payload

    def close(self):
        self.reader.close()
        self._chunk_cache.clear()
