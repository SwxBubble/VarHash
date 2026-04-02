#include "index/hamming_index.h"
#include "utils/hamming_distance.h"
#include <algorithm>
#include <fstream>
#include <limits>

namespace Delta {

namespace {
uint32_t ComputeDistance(const std::vector<uint64_t> &lhs,
                         const std::vector<uint64_t> &rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<uint32_t>::max();
  }
  uint32_t distance = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    distance += static_cast<uint32_t>(hammingDistance(lhs[i], rhs[i]));
  }
  return distance;
}
} // namespace

std::optional<chunk_id> HammingIndex::GetBaseChunkID(const Feature &feat) {
  auto ids = GetBaseChunkIDs(feat, 1);
  if (ids.empty()) {
    return std::nullopt;
  }
  return ids.front();
}

std::vector<chunk_id> HammingIndex::GetBaseChunkIDs(const Feature &feat,
                                                    size_t top_k) {
  const auto &hash_words = std::get<std::vector<uint64_t>>(feat);
  std::vector<std::pair<chunk_id, uint32_t>> ranked;
  ranked.reserve(entries_.size());
  for (const auto &[candidate_words, candidate_id] : entries_) {
    auto distance = ComputeDistance(hash_words, candidate_words);
    if (distance <= max_distance_) {
      ranked.push_back({candidate_id, distance});
    }
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second < rhs.second;
              }
              return lhs.first < rhs.first;
            });
  std::vector<chunk_id> result;
  result.reserve(std::min(top_k, ranked.size()));
  for (const auto &[candidate_id, distance] : ranked) {
    (void)distance;
    result.push_back(candidate_id);
    if (result.size() >= top_k) {
      break;
    }
  }
  return result;
}

void HammingIndex::AddFeature(const Feature &feat, chunk_id id) {
  entries_.push_back({std::get<std::vector<uint64_t>>(feat), id});
}

bool HammingIndex::DumpToFile(const std::string &path) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return false;
  }
  uint64_t entry_count = static_cast<uint64_t>(entries_.size());
  out.write(reinterpret_cast<const char *>(&max_distance_), sizeof(max_distance_));
  out.write(reinterpret_cast<const char *>(&entry_count), sizeof(entry_count));
  for (const auto &[words, id] : entries_) {
    uint64_t word_count = static_cast<uint64_t>(words.size());
    out.write(reinterpret_cast<const char *>(&word_count), sizeof(word_count));
    out.write(reinterpret_cast<const char *>(&id), sizeof(id));
    out.write(reinterpret_cast<const char *>(words.data()),
              static_cast<std::streamsize>(word_count * sizeof(uint64_t)));
  }
  return true;
}

bool HammingIndex::RecoverFromFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  entries_.clear();
  uint64_t entry_count = 0;
  in.read(reinterpret_cast<char *>(&max_distance_), sizeof(max_distance_));
  in.read(reinterpret_cast<char *>(&entry_count), sizeof(entry_count));
  for (uint64_t i = 0; i < entry_count; ++i) {
    uint64_t word_count = 0;
    chunk_id id = 0;
    in.read(reinterpret_cast<char *>(&word_count), sizeof(word_count));
    in.read(reinterpret_cast<char *>(&id), sizeof(id));
    std::vector<uint64_t> words(word_count, 0);
    in.read(reinterpret_cast<char *>(words.data()),
            static_cast<std::streamsize>(word_count * sizeof(uint64_t)));
    entries_.push_back({std::move(words), id});
  }
  return true;
}

} // namespace Delta
