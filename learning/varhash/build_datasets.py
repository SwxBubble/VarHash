import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def hash_split_value(value):
    checksum = sum(ord(ch) for ch in str(value))
    if checksum % 10 < 7:
        return "train"
    if checksum % 10 < 9:
        return "val"
    return "test"


def build_chronological_lookup(records):
    project_versions = defaultdict(set)
    for record in records:
        project_versions[record.get("project", "")].add(int(record.get("version_order", 0)))

    split_lookup = {}
    for project, versions in project_versions.items():
        ordered = sorted(versions)
        total = len(ordered)
        if total <= 2:
            for idx, version in enumerate(ordered):
                split_lookup[(project, version)] = "train" if idx == 0 else "test"
            continue
        train_end = max(1, int(total * 0.7))
        val_end = max(train_end + 1, int(total * 0.85))
        for idx, version in enumerate(ordered):
            if idx < train_end:
                split_lookup[(project, version)] = "train"
            elif idx < val_end:
                split_lookup[(project, version)] = "val"
            else:
                split_lookup[(project, version)] = "test"
    return split_lookup


def assign_split(record, mode, chrono_lookup):
    if mode == "chronological":
        key = (record.get("project", ""), int(record.get("version_order", 0)))
        return chrono_lookup.get(key, "train")
    if mode == "project":
        return hash_split_value(record.get("project", ""))
    if mode == "version":
        return hash_split_value(record.get("version", ""))
    return hash_split_value(record.get("timestamp", ""))


def is_exact_duplicate(record):
    pair_type = record.get("pair_type", "")
    return "exact_duplicate" in pair_type.split("|") or record.get("delta_size", -1) == 0


def classify_band(index, total):
    if total <= 2:
      return "head" if index == 0 else "tail"
    third = max(1, math.ceil(total / 3))
    if index < third:
        return "head"
    if index >= total - third:
        return "tail"
    return "middle"


def stratified_select(items, max_items, seed):
    if max_items <= 0 or len(items) <= max_items:
        return list(items)
    by_band = {"head": [], "middle": [], "tail": []}
    for item in items:
        by_band[item["band"]].append(item)

    quotas = {
        "head": max(1, max_items // 3),
        "middle": max(1, max_items // 3),
        "tail": max(1, max_items // 3),
    }
    selected = []
    rng = random.Random(seed)
    for band in ("head", "middle", "tail"):
        band_items = list(by_band[band])
        if len(band_items) <= quotas[band]:
            selected.extend(band_items)
            continue
        if band == "head":
            selected.extend(band_items[: quotas[band]])
        elif band == "tail":
            selected.extend(band_items[-quotas[band]:])
        else:
            rng.shuffle(band_items)
            selected.extend(band_items[: quotas[band]])

    if len(selected) >= max_items:
        return selected[:max_items]

    selected_ids = {(item["query_sha1"], item["ref_sha1"]) for item in selected}
    remaining = [item for item in items
                 if (item["query_sha1"], item["ref_sha1"]) not in selected_ids]
    remaining.sort(key=lambda item: item["retrieval_gain"], reverse=True)
    selected.extend(remaining[: max_items - len(selected)])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-by",
                        choices=["chronological", "project", "version", "timestamp"],
                        default="chronological")
    parser.add_argument("--positive-threshold", type=float, default=0.05)
    parser.add_argument("--max-candidates-per-query", type=int, default=32)
    parser.add_argument("--separate-exact-duplicates", action="store_true", default=True,
                        help="Write exact duplicates to exact_duplicate_{split}.jsonl and exclude them from the main training pairs")
    parser.add_argument("--no-separate-exact-duplicates", dest="separate_exact_duplicates",
                        action="store_false",
                        help="Disable exact duplicate side-file generation")
    parser.add_argument("--keep-exact-duplicates", action="store_true",
                        help="Keep exact duplicates in pair/query_group outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(load_candidates(args.candidates))
    chrono_lookup = build_chronological_lookup(records)

    pair_handles = {
        split: open(output_dir / f"pair_{split}.jsonl", "w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    exact_handles = None
    if args.separate_exact_duplicates:
        exact_handles = {
            split: open(output_dir / f"exact_duplicate_{split}.jsonl", "w", encoding="utf-8")
            for split in ("train", "val", "test")
        }

    grouped = defaultdict(list)
    for record in records:
        split = assign_split(record, args.split_by, chrono_lookup)
        record = dict(record)
        record["label"] = 1 if record["retrieval_gain"] >= args.positive_threshold else 0
        exact_dup = is_exact_duplicate(record)
        if exact_dup and exact_handles is not None:
            exact_handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        if exact_dup and not args.keep_exact_duplicates:
            continue
        pair_handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        grouped[(split, record["query_sha1"])].append(record)

    for handle in pair_handles.values():
        handle.close()
    if exact_handles is not None:
        for handle in exact_handles.values():
            handle.close()

    for split in ("train", "val", "test"):
        with open(output_dir / f"query_groups_{split}.jsonl", "w",
                  encoding="utf-8") as handle:
            for (group_split, query_sha1), items in grouped.items():
                if group_split != split:
                    continue
                items.sort(key=lambda item: item["retrieval_gain"], reverse=True)
                enriched = []
                for index, item in enumerate(items):
                    item = dict(item)
                    item["band"] = classify_band(index, len(items))
                    enriched.append(item)

                selected = stratified_select(
                    enriched,
                    args.max_candidates_per_query,
                    seed=sum(ord(ch) for ch in query_sha1),
                )
                selected.sort(key=lambda item: item["retrieval_gain"], reverse=True)

                positives = [
                    item["ref_sha1"] for item in selected
                    if item["retrieval_gain"] >= args.positive_threshold
                ]
                negatives = [
                    item["ref_sha1"] for item in selected
                    if item["retrieval_gain"] < args.positive_threshold
                ]
                head_refs = [item["ref_sha1"] for item in selected if item["band"] == "head"]
                middle_refs = [item["ref_sha1"] for item in selected if item["band"] == "middle"]
                tail_refs = [item["ref_sha1"] for item in selected if item["band"] == "tail"]
                middle_negatives = [
                    item["ref_sha1"] for item in selected
                    if item["band"] == "middle" and item["retrieval_gain"] < args.positive_threshold
                ]
                tail_negatives = [
                    item["ref_sha1"] for item in selected
                    if item["band"] == "tail" and item["retrieval_gain"] < args.positive_threshold
                ]
                hard_negatives = (middle_negatives + tail_negatives)[:8]
                if not selected:
                    continue
                group = {
                    "query_sha1": query_sha1,
                    "candidate_refs": [item["ref_sha1"] for item in selected],
                    "best_ref": selected[0]["ref_sha1"],
                    "best_gain": selected[0]["retrieval_gain"],
                    "positive_refs": positives,
                    "hard_negatives": hard_negatives if hard_negatives else negatives[:8],
                    "head_refs": head_refs,
                    "middle_refs": middle_refs,
                    "tail_refs": tail_refs,
                    "gain_rank": [
                        {
                            "ref_sha1": item["ref_sha1"],
                            "retrieval_gain": item["retrieval_gain"],
                            "pair_type": item.get("pair_type", ""),
                            "band": item["band"],
                        }
                        for item in selected
                    ],
                }
                handle.write(json.dumps(group, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
