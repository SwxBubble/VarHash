#include "index/best_fit_index.h"
#include <algorithm>
namespace Delta {
std::optional<chunk_id> BestFitIndex::GetBaseChunkID(const Feature &feat) {
  auto ids = GetBaseChunkIDs(feat, 1);
  if (ids.empty()) {
    return std::nullopt;
  }
  return ids.front();
}

std::vector<chunk_id> BestFitIndex::GetBaseChunkIDs(const Feature &feat,
                                                    size_t top_k) {
  const auto &features = std::get<std::vector<uint64_t>>(feat);
  std::unordered_map<chunk_id, uint32_t> match_count;
  for (int i = 0; i < feature_count_; i++) {
    const auto &index_i = index_[i];
    const uint64_t feature = features[i];
    if (!index_i.count(feature))
      continue;
    const auto &matched_chunk_ids = index_i.at(feature);
    for (const auto &id: matched_chunk_ids) {
      match_count[id]++;
    }
  }
  if (match_count.empty()) {
    return {};
  }
  std::vector<std::pair<chunk_id, uint32_t>> ranked(match_count.begin(),
                                                    match_count.end());
  std::sort(ranked.begin(), ranked.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });
  std::vector<chunk_id> result;
  result.reserve(std::min(top_k, ranked.size()));
  for (const auto &[id, count] : ranked) {
    if (count < 4) {
      continue;
    }
    result.push_back(id);
    if (result.size() >= top_k) {
      break;
    }
  }
  return result;
}

void BestFitIndex::AddFeature(const Feature &feat, chunk_id id) {
  const auto &features = std::get<std::vector<uint64_t>>(feat);
  for (int i = 0; i < feature_count_; i++) {
    index_[i][features[i]].push_back(id);
  }
}
} // namespace Delta
