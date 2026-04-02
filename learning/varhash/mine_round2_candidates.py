import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from model import VarHashNet


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_chunks(path):
    records = []
    for record in load_jsonl(path):
        record = dict(record)
        record["payload"] = base64.b64decode(record["payload_b64"])
        records.append(record)
    records.sort(key=lambda item: (
        item.get("project", ""),
        int(item.get("version_order", 0)),
        int(item.get("chunk_offset", 0)),
        item["sha1"],
    ))
    return records


def encode_payload(payload, max_length=None):
    raw = torch.tensor(list(payload), dtype=torch.float32)
    if max_length is None:
        max_length = raw.numel()
    padded = torch.zeros(max_length, dtype=torch.float32)
    padded[: raw.numel()] = raw / 127.5 - 1.0
    return padded, raw.numel()


def batch_encode(records, model, device, batch_size):
    embeddings = {}
    start = 0
    while start < len(records):
        batch_records = records[start:start + batch_size]
        max_len = max(len(item["payload"]) for item in batch_records)
        tensors = []
        lengths = []
        for item in batch_records:
            tensor, length = encode_payload(item["payload"], max_length=max_len)
            tensors.append(tensor)
            lengths.append(length)
        inputs = torch.stack(tensors, dim=0).to(device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        with torch.no_grad():
            embed, _ = model(inputs, lengths)
            embed = F.normalize(embed, dim=-1)
        for item, vector in zip(batch_records, embed.cpu()):
            embeddings[item["sha1"]] = vector
        start += batch_size
    return embeddings


def load_existing_pairs(path):
    if not path:
        return set()
    pairs = set()
    for record in load_jsonl(path):
        pairs.add((record["query_sha1"], record["ref_sha1"]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--existing-candidates", default="",
                        help="Optional candidate_pairs.jsonl to avoid re-emitting known pairs")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-history-version-gap", type=int, default=1)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = VarHashNet(hash_bits=checkpoint["hash_bits"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    records = load_chunks(args.chunks)
    embeddings = batch_encode(records, model, device, args.batch_size)
    existing_pairs = load_existing_pairs(args.existing_candidates)

    grouped = defaultdict(list)
    for record in records:
        grouped[record["project"]].append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for project, project_records in grouped.items():
            history = []
            current_version = None
            pending = []

            def flush_pending():
                history.extend(pending)

            for record in project_records:
                version_order = int(record.get("version_order", 0))
                if current_version is None:
                    current_version = version_order
                if version_order != current_version:
                    flush_pending()
                    pending = []
                    current_version = version_order

                valid_history = [
                    item for item in history
                    if int(item.get("version_order", 0)) <= version_order - args.min_history_version_gap
                ]
                if valid_history:
                    q_vec = embeddings[record["sha1"]]
                    scored = []
                    for ref in valid_history:
                        if (record["sha1"], ref["sha1"]) in existing_pairs:
                            continue
                        score = torch.dot(q_vec, embeddings[ref["sha1"]]).item()
                        scored.append((score, ref))
                    scored.sort(key=lambda item: item[0], reverse=True)
                    for rank, (score, ref) in enumerate(scored[: args.top_k], start=1):
                        row = {
                            "query_sha1": record["sha1"],
                            "ref_sha1": ref["sha1"],
                            "project": project,
                            "version": record["version"],
                            "version_order": version_order,
                            "query_chunk_id": record.get("chunk_id"),
                            "ref_chunk_id": ref.get("chunk_id"),
                            "query_offset": record.get("chunk_offset"),
                            "ref_offset": ref.get("chunk_offset"),
                            "pair_type": "model_mined",
                            "rank": rank,
                            "score": score,
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")

                pending.append(record)


if __name__ == "__main__":
    main()
