# VarHash integration

This repository now includes a `varhash` feature path for variable-size
FastCDC chunks.

## Online path

Set `feature.type = "varhash"` in a config file and run:

```bash
./build/delta --config=varhash.toml
```

`VarHashFeature` supports two modes:

- heuristic mode: derive a multi-view binary hash directly from chunk bytes
- precomputed mode: load offline semantic hashes from `precomputed_hash_path`

The online index uses Hamming distance and returns the nearest reference when
the distance is below `max_hamming_distance`.

`top_k_candidates` controls how many references are recalled from the index.
The compression path then re-ranks those candidates with real xdelta output
size and only uses delta encoding when the byte gain is at least
`storage.min_gain_bytes`.

## Offline training path

See `learning/varhash/` for scripts to:

- build `pair_*.jsonl` and `query_groups_*.jsonl`
- train a byte-level semantic hash model
- export precomputed hashes for the C++ framework

The exported file format is:

```text
<sha1hex> <word0_hex> <word1_hex> ...
```

Each `wordN_hex` is a 64-bit hash word in hexadecimal.
