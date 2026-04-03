import argparse
import atexit
import json
import multiprocessing as mp
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path

from chunk_store import TarChunkReader, load_grouped_chunk_records, load_jsonl


_WORKER_READER = None
_WORKER_XDELTA3 = None
_WORKER_BASE_PATH = None
_WORKER_QUERY_PATH = None
_WORKER_DELTA_PATH = None
_WORKER_TEMP_DIR = None


def load_external_candidates(path, default_pair_type):
    candidates = defaultdict(list)
    if not path:
        return candidates
    for record in load_jsonl(path):
        query_sha1 = record.get("query_sha1")
        ref_sha1 = record.get("ref_sha1")
        if not query_sha1 or not ref_sha1:
            continue
        row = dict(record)
        row.setdefault("pair_type", default_pair_type)
        candidates[query_sha1].append(row)
    return candidates


def append_candidate(candidate_map, query_record, ref_record, pair_type):
    if ref_record["chunk_id"] == query_record["chunk_id"]:
        return
    key = ref_record["chunk_id"]
    if key not in candidate_map:
        candidate_map[key] = {"ref": ref_record, "pair_types": [pair_type]}
        return
    if pair_type not in candidate_map[key]["pair_types"]:
        candidate_map[key]["pair_types"].append(pair_type)


def resolve_external_refs(external_records, history_by_sha1):
    resolved = []
    for record in external_records:
        matches = history_by_sha1.get(record["ref_sha1"], [])
        if matches:
            resolved.append((matches[-1], record.get("pair_type", "external_candidate")))
    return resolved


def collect_candidates_for_query(query_record, history_by_sha1, history_by_bucket,
                                 history_all, external_candidates, args, rng):
    candidates = {}

    for ref_record in history_by_sha1.get(query_record["sha1"], [])[:args.exact_dups]:
        append_candidate(candidates, query_record, ref_record, "exact_duplicate")

    bucket_id = int(query_record.get("chunk_offset", 0)) // args.bucket_width
    bucket_records = []
    for neighbor_bucket in range(bucket_id - 1, bucket_id + 2):
        bucket_records.extend(history_by_bucket.get(neighbor_bucket, []))

    if args.recent_version_window > 0:
        min_version = int(query_record.get("version_order", 0)) - args.recent_version_window
        bucket_records = [
            item for item in bucket_records
            if int(item.get("version_order", 0)) >= min_version
        ]

    bucket_records.sort(
        key=lambda item: abs(
            int(item.get("chunk_offset", 0)) - int(query_record.get("chunk_offset", 0))
        )
    )
    for ref_record in bucket_records[:args.offset_neighbors]:
        append_candidate(candidates, query_record, ref_record, "offset_neighbor")

    if args.random_negatives > 0 and history_all:
        sample_size = min(args.random_negatives, len(history_all))
        for ref_record in rng.sample(history_all, sample_size):
            append_candidate(candidates, query_record, ref_record, "random_history")

    for ref_record, pair_type in external_candidates:
        append_candidate(candidates, query_record, ref_record, pair_type)

    return list(candidates.values())


def select_candidate_subset(candidates, max_candidates_per_query, rng):
    if max_candidates_per_query <= 0 or len(candidates) <= max_candidates_per_query:
        return list(candidates)

    exact = [c for c in candidates if "exact_duplicate" in c["pair_types"]]
    bootstrap = [
        c for c in candidates
        if any(tag in c["pair_types"] for tag in ("odess_topk", "finesse_topk"))
    ]
    offset = [c for c in candidates if "offset_neighbor" in c["pair_types"]]
    random_hist = [c for c in candidates if "random_history" in c["pair_types"]]

    selected = []
    seen = set()

    def take(items, limit, shuffle=False):
        pool = list(items)
        if shuffle:
            rng.shuffle(pool)
        for item in pool[:limit]:
            ref_id = item["ref"]["chunk_id"]
            if ref_id not in seen:
                seen.add(ref_id)
                selected.append(item)

    quota = max(1, max_candidates_per_query // 4)
    take(exact, min(len(exact), max(1, quota // 2)))
    take(bootstrap, min(len(bootstrap), quota))
    take(offset, min(len(offset), quota))
    take(random_hist, min(len(random_hist), quota), shuffle=True)

    if len(selected) < max_candidates_per_query:
        remaining = [c for c in candidates if c["ref"]["chunk_id"] not in seen]
        rng.shuffle(remaining)
        take(remaining, max_candidates_per_query - len(selected))

    return selected[:max_candidates_per_query]


def trim_history_if_needed(history_all, history_queue, history_by_sha1, history_by_bucket,
                           max_history_per_project):
    if max_history_per_project <= 0:
        return

    while len(history_all) > max_history_per_project and history_queue:
        removed = history_queue.popleft()
        removed_id = removed["chunk_id"]

        history_all.pop(removed_id, None)

        sha1_bucket = history_by_sha1.get(removed["sha1"], [])
        history_by_sha1[removed["sha1"]] = [
            item for item in sha1_bucket if item["chunk_id"] != removed_id
        ]
        if not history_by_sha1[removed["sha1"]]:
            del history_by_sha1[removed["sha1"]]

        bucket_id = int(removed.get("chunk_offset", 0)) // bucket_width_from_record(removed)
        bucket_records = history_by_bucket.get(bucket_id, [])
        history_by_bucket[bucket_id] = [
            item for item in bucket_records if item["chunk_id"] != removed_id
        ]
        if not history_by_bucket[bucket_id]:
            del history_by_bucket[bucket_id]


def bucket_width_from_record(record):
    return int(record.get("_bucket_width", 8192))


def add_history_record(record, history_all, history_queue, history_by_sha1, history_by_bucket, bucket_width):
    record = dict(record)
    record["_bucket_width"] = bucket_width
    history_all[record["chunk_id"]] = record
    history_queue.append(record)
    history_by_sha1[record["sha1"]].append(record)
    bucket_id = int(record.get("chunk_offset", 0)) // bucket_width
    history_by_bucket[bucket_id].append(record)


def build_query_tasks(grouped, odess_candidates, finesse_candidates, args):
    tasks = []
    processed_queries = 0
    task_rng = random.Random(args.seed)

    for project, records in grouped.items():
        version_groups = defaultdict(list)
        for record in records:
            version_groups[int(record.get("version_order", 0))].append(record)
        ordered_versions = sorted(version_groups)

        history_all = {}
        history_queue = deque()
        history_by_sha1 = defaultdict(list)
        history_by_bucket = defaultdict(list)
        history_cursor = 0

        for version_order in ordered_versions:
            eligible_cutoff = version_order - args.min_history_version_gap
            while history_cursor < len(ordered_versions) and ordered_versions[history_cursor] <= eligible_cutoff:
                history_version = ordered_versions[history_cursor]
                for item in version_groups[history_version]:
                    add_history_record(
                        item,
                        history_all,
                        history_queue,
                        history_by_sha1,
                        history_by_bucket,
                        args.bucket_width,
                    )
                if args.max_history_per_project > 0:
                    trim_history_if_needed(
                        history_all,
                        history_queue,
                        history_by_sha1,
                        history_by_bucket,
                        args.max_history_per_project,
                    )
                history_cursor += 1

            if not history_all:
                continue

            history_values = list(history_all.values())
            for record in version_groups[version_order]:
                if args.max_queries > 0 and processed_queries >= args.max_queries:
                    return tasks

                external_for_query = []
                external_for_query.extend(
                    resolve_external_refs(odess_candidates.get(record["sha1"], []), history_by_sha1)
                )
                external_for_query.extend(
                    resolve_external_refs(finesse_candidates.get(record["sha1"], []), history_by_sha1)
                )

                candidates = collect_candidates_for_query(
                    record,
                    history_by_sha1,
                    history_by_bucket,
                    history_values,
                    external_for_query,
                    args,
                    random.Random(task_rng.randint(0, 2**31 - 1)),
                )
                selected = select_candidate_subset(
                    candidates,
                    args.max_candidates_per_query,
                    rng=random.Random(task_rng.randint(0, 2**31 - 1)),
                )
                if not selected:
                    continue

                tasks.append({
                    "query": record,
                    "candidates": selected,
                })
                processed_queries += 1

    return tasks


def resolve_xdelta3_path(xdelta3_path):
    if Path(xdelta3_path).exists():
        return str(Path(xdelta3_path))
    resolved = shutil.which(xdelta3_path)
    if resolved:
        return resolved
    raise RuntimeError(
        f"xdelta3 binary '{xdelta3_path}' was not found. Install xdelta3 or pass --xdelta3."
    )


def cleanup_worker():
    global _WORKER_READER, _WORKER_TEMP_DIR
    if _WORKER_READER is not None:
        _WORKER_READER.close()
        _WORKER_READER = None
    if _WORKER_TEMP_DIR is not None:
        _WORKER_TEMP_DIR.cleanup()
        _WORKER_TEMP_DIR = None


def init_worker(tar_root, xdelta3_path, max_open_tar_files):
    global _WORKER_READER, _WORKER_XDELTA3
    global _WORKER_BASE_PATH, _WORKER_QUERY_PATH, _WORKER_DELTA_PATH, _WORKER_TEMP_DIR

    _WORKER_READER = TarChunkReader(tar_root, max_open_files=max_open_tar_files)
    _WORKER_XDELTA3 = xdelta3_path
    _WORKER_TEMP_DIR = tempfile.TemporaryDirectory(prefix="varhash_xdelta_")
    temp_dir = Path(_WORKER_TEMP_DIR.name)
    _WORKER_BASE_PATH = temp_dir / "base.bin"
    _WORKER_QUERY_PATH = temp_dir / "query.bin"
    _WORKER_DELTA_PATH = temp_dir / "delta.bin"
    atexit.register(cleanup_worker)


def compute_delta_size(base_payload, query_payload):
    _WORKER_BASE_PATH.write_bytes(base_payload)
    _WORKER_QUERY_PATH.write_bytes(query_payload)
    completed = subprocess.run(
        [
            _WORKER_XDELTA3,
            "-f",
            "-e",
            "-s",
            str(_WORKER_BASE_PATH),
            str(_WORKER_QUERY_PATH),
            str(_WORKER_DELTA_PATH),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"xdelta3 failed with code {completed.returncode}: {completed.stderr.strip()}"
        )
    return _WORKER_DELTA_PATH.stat().st_size


def process_query_task(task):
    query = task["query"]
    query_payload = _WORKER_READER.read_record(query)
    rows = []
    query_length = int(query["chunk_length"])

    for candidate in task["candidates"]:
        ref = candidate["ref"]
        ref_payload = _WORKER_READER.read_record(ref)
        delta_size = compute_delta_size(ref_payload, query_payload)
        retrieval_gain = (query_length - delta_size) / max(query_length, 1)
        rows.append({
            "query_sha1": query["sha1"],
            "ref_sha1": ref["sha1"],
            "project": query["project"],
            "version": query["version"],
            "version_order": int(query["version_order"]),
            "query_chunk_id": query["chunk_id"],
            "ref_chunk_id": ref["chunk_id"],
            "query_offset": query["chunk_offset"],
            "ref_offset": ref["chunk_offset"],
            "query_length": query_length,
            "ref_length": int(ref["chunk_length"]),
            "delta_size": delta_size,
            "retrieval_gain": retrieval_gain,
            "pair_type": "|".join(candidate["pair_types"]),
        })
    return rows


def parse_projects(value):
    return {item.strip() for item in value.split(",") if item.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="chunks.jsonl path")
    parser.add_argument("--tar-root", required=True,
                        help="Root directory that contains the tar files referenced by tar_path")
    parser.add_argument("--output", required=True, help="candidate_pairs.jsonl path")
    parser.add_argument("--xdelta3", default="xdelta3", help="xdelta3 binary path")
    parser.add_argument("--bucket-width", type=int, default=8192)
    parser.add_argument("--exact-dups", type=int, default=4)
    parser.add_argument("--offset-neighbors", type=int, default=12)
    parser.add_argument("--random-negatives", type=int, default=8)
    parser.add_argument("--odess-candidates", default="", help="Optional Odess top-k jsonl")
    parser.add_argument("--finesse-candidates", default="", help="Optional Finesse top-k jsonl")
    parser.add_argument("--max-history-per-project", type=int, default=0)
    parser.add_argument("--min-history-version-gap", type=int, default=1)
    parser.add_argument("--recent-version-window", type=int, default=2)
    parser.add_argument("--projects", default="", help="Comma-separated project filter")
    parser.add_argument("--min-version-order", type=int, default=None)
    parser.add_argument("--max-version-order", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=0,
                        help="If >0, only label the first N valid queries after filtering")
    parser.add_argument("--max-candidates-per-query", type=int, default=16,
                        help="Budgeted number of candidates to label per query; 0 keeps all")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for xdelta labeling")
    parser.add_argument("--max-open-tar-files", type=int, default=8,
                        help="Per-process tar handle cache size")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    xdelta3_path = resolve_xdelta3_path(args.xdelta3)
    allowed_projects = parse_projects(args.projects)
    grouped, _ = load_grouped_chunk_records(
        args.chunks,
        allowed_projects=allowed_projects,
        min_version_order=args.min_version_order,
        max_version_order=args.max_version_order,
    )
    odess_candidates = load_external_candidates(args.odess_candidates, "odess_topk")
    finesse_candidates = load_external_candidates(args.finesse_candidates, "finesse_topk")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = build_query_tasks(grouped, odess_candidates, finesse_candidates, args)
    print(f"prepared_queries={len(tasks)} workers={args.workers}")

    with output_path.open("w", encoding="utf-8") as out:
        if args.workers <= 1:
            init_worker(args.tar_root, xdelta3_path, args.max_open_tar_files)
            try:
                iterator = map(process_query_task, tasks)
                for index, rows in enumerate(iterator, start=1):
                    for row in rows:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if index % 100 == 0:
                        print(f"processed_queries={index}/{len(tasks)}")
            finally:
                cleanup_worker()
        else:
            pool_size = min(args.workers, mp.cpu_count())
            with mp.Pool(
                processes=pool_size,
                initializer=init_worker,
                initargs=(args.tar_root, xdelta3_path, args.max_open_tar_files),
            ) as pool:
                for index, rows in enumerate(
                    pool.imap_unordered(process_query_task, tasks, chunksize=1),
                    start=1,
                ):
                    for row in rows:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if index % 100 == 0:
                        print(f"processed_queries={index}/{len(tasks)}")


if __name__ == "__main__":
    main()
