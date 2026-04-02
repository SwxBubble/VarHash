#include "index/palantir_index.h"
#include <unordered_set>
namespace Delta {
std::optional<chunk_id> PalantirIndex::GetBaseChunkID(const Feature &feat) {
  auto ids = GetBaseChunkIDs(feat, 1);
  if (ids.empty()) {
    return std::nullopt;
  }
  return ids.front();
}

std::vector<chunk_id> PalantirIndex::GetBaseChunkIDs(const Feature &feat,
                                                     size_t top_k) {
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  std::vector<chunk_id> result;
  std::unordered_set<chunk_id> seen;
  for (int i = 0; i < features.size(); i++) {
    auto level_ids = levels_[i]->GetBaseChunkIDs(features[i], top_k);
    for (auto id : level_ids) {
      if (seen.insert(id).second) {
        result.push_back(id);
        if (result.size() >= top_k) {
          return result;
        }
      }
    }
  }
  return result;
}

void PalantirIndex::AddFeature(const Feature &feat, chunk_id id) {
  const auto &features = std::get<std::vector<std::vector<uint64_t>>>(feat);
  for (int i = 0; i < features.size(); i++) {
    levels_[i]->AddFeature(features[i], id);
  }
}
} // namespace Delta
