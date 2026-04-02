#pragma once
#include "index/index.h"
#include <utility>
#include <vector>
namespace Delta {

class HammingIndex : public Index {
public:
  explicit HammingIndex(uint32_t max_distance = 24)
      : max_distance_(max_distance) {}

  std::optional<chunk_id> GetBaseChunkID(const Feature &feat) override;
  std::vector<chunk_id> GetBaseChunkIDs(const Feature &feat,
                                        size_t top_k) override;
  void AddFeature(const Feature &feat, chunk_id id) override;
  bool RecoverFromFile(const std::string &path) override;
  bool DumpToFile(const std::string &path) override;

private:
  std::vector<std::pair<std::vector<uint64_t>, chunk_id>> entries_;
  uint32_t max_distance_;
};

} // namespace Delta
