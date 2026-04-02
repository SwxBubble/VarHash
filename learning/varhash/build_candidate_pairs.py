import argparse
import base64
import json
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
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


def build_grouped_records(chunks_path):
    grouped = defaultdict(list)
    for record in load_jsonl(chunks_path):
        record["payload"] = base64.b64decode(record["payload_b64"])
        grouped[record["project"]].append(record)
    for project in grouped:
        grouped[project].sort(key=chunk_sort_key)
    return grouped


def load_external_candidates(path, default_pair_type):
    candidates = defaultdict(list)
    if not path:
        return candidates
    for record in load_jsonl(path):
        query_sha1 = record.get("query_sha1")
        ref_sha1 = record.get("ref_sha1")
        if not query_sha1 or not ref_sha1:
            continue
        record = dict(record)
        record.setdefault("pair_type", default_pair_type)
        candidates[query_sha1].append(record)
    return candidates


def compute_delta_size(base_payload, query_payload, xdelta3_path):
    if shutil.which(xdelta3_path) is None and not Path(xdelta3_path).exists():
        raise RuntimeError(
            f"xdelta3 binary '{xdelta3_path}' was not found. Please install xdelta3 or pass --xdelta3."
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "base.bin"
        query_path = Path(tmpdir) / "query.bin"
        delta_path = Path(tmpdir) / "delta.bin"
        base_path.write_bytes(base_payload)
        query_path.write_bytes(query_payload)
        cmd = [xdelta3_path, "-f", "-e", "-s", str(base_path), str(query_path), str(delta_path)]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"xdelta3 failed with code {completed.returncode}: {completed.stderr.strip()}"
            )
        return delta_path.stat().st_size


def append_candidate(candidate_map, query_record, ref_record, pair_type):
    if ref_record["chunk_id"] == query_record["chunk_id"]:
        return
    key = ref_record["chunk_id"]
    if key not in candidate_map:
        candidate_map[key] = {
            "ref": ref_record,
            "pair_types": [pair_type],
        }
        return
    if pair_type not in candidate_map[key]["pair_types"]:
        candidate_map[key]["pair_types"].append(pair_type)


def resolve_external_refs(query_record, external_records, history_by_sha1):
    resolved = []
    for record in external_records:
        ref_sha1 = record["ref_sha1"]
        history_matches = history_by_sha1.get(ref_sha1, [])
        if not history_matches:
            continue
        resolved.append((history_matches[-1], record.get("pair_type", "external_candidate")))
    return resolved


def collect_candidates_for_query(query_record, history_by_sha1, history_by_bucket,
                                 history_all, external_candidates, exact_dups,
                                 offset_neighbors, random_negatives, bucket_width,
                                 recent_version_window):
    candidates = {}

    for ref_record in history_by_sha1.get(query_record["sha1"], [])[:exact_dups]:
        append_candidate(candidates, query_record, ref_record, "exact_duplicate")

    bucket_id = int(query_record.get("chunk_offset", 0)) // bucket_width
    bucket_records = []
    for neighbor_bucket in range(bucket_id - 1, bucket_id + 2):
        bucket_records.extend(history_by_bucket.get(neighbor_bucket, []))
    if recent_version_window > 0:
        min_version = int(query_record.get("version_order", 0)) - recent_version_window
        bucket_records = [
            item for item in bucket_records
            if int(item.get("version_order", 0)) >= min_version
        ]
    bucket_records.sort(
        key=lambda item: abs(int(item.get("chunk_offset", 0)) - int(query_record.get("chunk_offset", 0)))
    )
    for ref_record in bucket_records[:offset_neighbors]:
        append_candidate(candidates, query_record, ref_record, "offset_neighbor")

    if random_negatives > 0 and history_all:
        sample_size = min(random_negatives, len(history_all))
        for ref_record in random.sample(history_all, sample_size):
            append_candidate(candidates, query_record, ref_record, "random_history")

    for ref_record, pair_type in external_candidates:
        append_candidate(candidates, query_record, ref_record, pair_type)

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="chunks.jsonl path")
    parser.add_argument("--output", required=True, help="candidate_pairs.jsonl path")
    parser.add_argument("--xdelta3", default="xdelta3", help="xdelta3 binary path")
    parser.add_argument("--bucket-width", type=int, default=8192)
    parser.add_argument("--exact-dups", type=int, default=4)
    parser.add_argument("--offset-neighbors", type=int, default=12)
    parser.add_argument("--random-negatives", type=int, default=8)
    parser.add_argument("--odess-candidates", default="",
                        help="Optional jsonl file with Odess top-k candidates")
    parser.add_argument("--finesse-candidates", default="",
                        help="Optional jsonl file with Finesse top-k candidates")
    parser.add_argument("--max-history-per-project", type=int, default=0,
                        help="If >0, only keep the most recent N history chunks per project in candidate generation")
    parser.add_argument("--min-history-version-gap", type=int, default=1,
                        help="Require ref.version_order <= query.version_order - gap")
    parser.add_argument("--recent-version-window", type=int, default=2,
                        help="Offset-neighbor candidates only use the most recent N historical versions; 0 disables the window")
    args = parser.parse_args()

    grouped = build_grouped_records(args.chunks)
    odess_candidates = load_external_candidates(args.odess_candidates, "odess_topk")
    finesse_candidates = load_external_candidates(args.finesse_candidates, "finesse_topk")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for project, records in grouped.items():
            history_all = []
            history_by_sha1 = defaultdict(list)
            history_by_bucket = defaultdict(list)
            current_version = None
            pending_version_records = []

            def flush_pending():
                for record in pending_version_records:
                    history_all.append(record)
                    history_by_sha1[record["sha1"]].append(record)
                    bucket_id = int(record.get("chunk_offset", 0)) // args.bucket_width
                    history_by_bucket[bucket_id].append(record)
                if args.max_history_per_project > 0 and len(history_all) > args.max_history_per_project:
                    overflow = len(history_all) - args.max_history_per_project
                    removed = history_all[:overflow]
                    del history_all[:overflow]
                    removed_ids = {item["chunk_id"] for item in removed}
                    for sha1_hex, bucket_records in list(history_by_sha1.items()):
                        history_by_sha1[sha1_hex] = [
                            item for item in bucket_records if item["chunk_id"] not in removed_ids
                        ]
                        if not history_by_sha1[sha1_hex]:
                            del history_by_sha1[sha1_hex]
                    for bucket, bucket_records in list(history_by_bucket.items()):
                        history_by_bucket[bucket] = [
                            item for item in bucket_records if item["chunk_id"] not in removed_ids
                        ]
                        if not history_by_bucket[bucket]:
                            del history_by_bucket[bucket]

            for record in records:
                version_order = int(record.get("version_order", 0))
                if current_version is None:
                    current_version = version_order
                if version_order != current_version:
                    flush_pending()
                    pending_version_records = []
                    current_version = version_order

                if not history_all:
                    pending_version_records.append(record)
                    continue

                valid_history = [
                    item for item in history_all
                    if int(item.get("version_order", 0)) <= version_order - args.min_history_version_gap
                ]
                if not valid_history:
                    pending_version_records.append(record)
                    continue

                valid_sha1 = defaultdict(list)
                valid_bucket = defaultdict(list)
                for item in valid_history:
                    valid_sha1[item["sha1"]].append(item)
                    bucket_id = int(item.get("chunk_offset", 0)) // args.bucket_width
                    valid_bucket[bucket_id].append(item)

                external_for_query = []
                external_for_query.extend(
                    resolve_external_refs(record, odess_candidates.get(record["sha1"], []), valid_sha1)
                )
                external_for_query.extend(
                    resolve_external_refs(record, finesse_candidates.get(record["sha1"], []), valid_sha1)
                )

                candidates = collect_candidates_for_query(
                    record,
                    valid_sha1,
                    valid_bucket,
                    valid_history,
                    external_for_query,
                    args.exact_dups,
                    args.offset_neighbors,
                    args.random_negatives,
                    args.bucket_width,
                    args.recent_version_window,
                )

                for candidate in candidates.values():
                    ref_record = candidate["ref"]
                    delta_size = compute_delta_size(
                        ref_record["payload"], record["payload"], args.xdelta3
                    )
                    retrieval_gain = (
                        (len(record["payload"]) - delta_size) / max(len(record["payload"]), 1)
                    )
                    row = {
                        "query_sha1": record["sha1"],
                        "ref_sha1": ref_record["sha1"],
                        "project": project,
                        "version": record["version"],
                        "version_order": version_order,
                        "query_chunk_id": record["chunk_id"],
                        "ref_chunk_id": ref_record["chunk_id"],
                        "query_offset": record["chunk_offset"],
                        "ref_offset": ref_record["chunk_offset"],
                        "query_length": len(record["payload"]),
                        "ref_length": len(ref_record["payload"]),
                        "delta_size": delta_size,
                        "retrieval_gain": retrieval_gain,
                        "pair_type": "|".join(candidate["pair_types"]),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")

                pending_version_records.append(record)


if __name__ == "__main__":
    main()
