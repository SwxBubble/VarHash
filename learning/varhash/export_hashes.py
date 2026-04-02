import argparse
import base64
import json
from pathlib import Path

import torch

from model import VarHashNet


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = VarHashNet(hash_bits=checkpoint["hash_bits"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in load_jsonl(args.chunks):
            payload = base64.b64decode(record["payload_b64"])
            tensor, lengths = bytes_to_tensor(payload)
            with torch.no_grad():
                _, hash_tensor = model(tensor, lengths)
            words = pack_bits(hash_tensor)
            handle.write(record["sha1"] + " " + " ".join(words) + "\n")


if __name__ == "__main__":
    main()
