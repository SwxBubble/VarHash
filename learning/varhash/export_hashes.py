import argparse
from pathlib import Path

import torch

from chunk_store import ChunkStore, load_jsonl
from model import VarHashNet


def bytes_to_tensor(payload):
    raw = torch.tensor(list(payload), dtype=torch.float32)
    padded = raw / 127.5 - 1.0
    return padded.unsqueeze(0), torch.tensor([raw.numel()], dtype=torch.long)


def pack_bits(hash_tensor):
    binary = (hash_tensor.squeeze(0) >= 0).to(torch.int64).tolist()
    words = []
    for start in range(0, len(binary), 64):
        value = 0
        for offset, bit in enumerate(binary[start:start + 64]):
            if bit:
                value |= 1 << offset
        words.append(f"{value:016x}")
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--tar-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-open-tar-files", type=int, default=8)
    parser.add_argument("--max-cached-chunks", type=int, default=4096)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = VarHashNet(hash_bits=checkpoint["hash_bits"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_store = ChunkStore(
        args.chunks,
        args.tar_root,
        max_open_files=args.max_open_tar_files,
        max_cached_chunks=args.max_cached_chunks,
    )
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for record in load_jsonl(args.chunks):
                payload = chunk_store.get(record["sha1"])
                tensor, lengths = bytes_to_tensor(payload)
                with torch.no_grad():
                    _, hash_tensor = model(tensor, lengths)
                words = pack_bits(hash_tensor)
                handle.write(record["sha1"] + " " + " ".join(words) + "\n")
    finally:
        chunk_store.close()


if __name__ == "__main__":
    main()
