import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chunk_store import ChunkStore, load_jsonl
from model import VarHashNet


class QueryGroupDataset(Dataset):
    def __init__(self, chunk_store, query_groups_path):
        self.chunk_store = chunk_store
        self.groups = list(load_jsonl(query_groups_path))
        self.groups = [group for group in self.groups
                       if group.get("best_ref") and group.get("hard_negatives")]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        negative = random.choice(group["hard_negatives"])
        return (
            self.chunk_store.get(group["query_sha1"]),
            self.chunk_store.get(group["best_ref"]),
            self.chunk_store.get(negative),
        )


def pack_batch(batch):
    tensors = []
    lengths = []
    max_len = max(len(item) for triple in batch for item in triple)
    for triple in batch:
        packed = []
        triple_lengths = []
        for item in triple:
            raw = torch.tensor(list(item), dtype=torch.float32)
            triple_lengths.append(raw.numel())
            padded = torch.zeros(max_len, dtype=torch.float32)
            padded[: raw.numel()] = raw / 127.5 - 1.0
            packed.append(padded)
        tensors.append(torch.stack(packed, dim=0))
        lengths.append(torch.tensor(triple_lengths, dtype=torch.long))
    return torch.stack(tensors, dim=0), torch.stack(lengths, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--tar-root", required=True)
    parser.add_argument("--query-groups", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hash-bits", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-open-tar-files", type=int, default=8)
    parser.add_argument("--max-cached-chunks", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_store = ChunkStore(
        args.chunks,
        args.tar_root,
        max_open_files=args.max_open_tar_files,
        max_cached_chunks=args.max_cached_chunks,
    )
    dataset = QueryGroupDataset(chunk_store, args.query_groups)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=pack_batch, num_workers=args.num_workers)

    model = VarHashNet(hash_bits=args.hash_bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    try:
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch, lengths in loader:
                batch = batch.to(device)
                lengths = lengths.to(device)
                query = batch[:, 0, :]
                positive = batch[:, 1, :]
                negative = batch[:, 2, :]
                query_len = lengths[:, 0]
                positive_len = lengths[:, 1]
                negative_len = lengths[:, 2]

                q_embed, q_hash = model(query, query_len)
                p_embed, p_hash = model(positive, positive_len)
                n_embed, n_hash = model(negative, negative_len)

                pos_score = F.cosine_similarity(q_embed, p_embed)
                neg_score = F.cosine_similarity(q_embed, n_embed)
                rank_loss = F.relu(args.margin - pos_score + neg_score).mean()
                quant_loss = (
                    (q_hash.abs() - 1.0).abs().mean()
                    + (p_hash.abs() - 1.0).abs().mean()
                    + (n_hash.abs() - 1.0).abs().mean()
                )
                bit_balance_loss = (
                    q_hash.mean(dim=0).abs().mean()
                    + p_hash.mean(dim=0).abs().mean()
                    + n_hash.mean(dim=0).abs().mean()
                )
                loss = rank_loss + 0.1 * quant_loss + 0.05 * bit_balance_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            print(f"epoch={epoch} loss={total_loss / max(len(loader), 1):.4f}")

        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "hash_bits": args.hash_bits},
                   args.checkpoint)
    finally:
        chunk_store.close()


if __name__ == "__main__":
    main()
