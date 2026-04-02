#include "chunk/chunk.h"
#include "index/super_feature_index.h"
#include "feature/features.h"
#include <assert.h>
#include <fstream>
#include <glog/logging.h>
#include <algorithm>
namespace Delta {
using chunk_id = uint32_t;
std::optional<chunk_id>
SuperFeatureIndex::GetBaseChunkID(const Feature &feat) {
  auto ids = GetBaseChunkIDs(feat, 1);
  if (ids.empty()) {
    return std::nullopt;
  }
  return ids.front();
}

std::vector<chunk_id> SuperFeatureIndex::GetBaseChunkIDs(const Feature &feat,
                                                         size_t top_k) {
  const auto &super_feature = std::get<std::vector<uint64_t>>(feat);
  std::unordered_map<chunk_id, uint32_t> match_count;
  for (int i = 0; i < super_feature_count_; i++) {
    if (!index_[i].count(super_feature[i])) {
      continue;
    }
    match_count[index_[i][super_feature[i]]]++;
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
    if (count == 0) {
      continue;
    }
    result.push_back(id);
    if (result.size() >= top_k) {
      break;
    }
  }
  return result;
}

void SuperFeatureIndex::AddFeature(const Feature &feat, chunk_id id) {
  const auto &super_feature = std::get<std::vector<uint64_t>>(feat);
  for (int i = 0; i < super_feature_count_; i++) {
    index_[i][super_feature[i]] = id;
  }
}

bool SuperFeatureIndex::DumpToFile(const std::string &path) {
  std::ofstream outFile(path, std::ios::binary);
  if (!outFile) {
    LOG(FATAL) << "SuperFeatureIndex::DumpToFile: cannot open output file "
              << path;
    return false;
  }
  auto write_uint64 = [&](uint64_t data) {
    outFile.write(reinterpret_cast<const char *>(&data), sizeof(uint64_t));
  };
  auto write_uint32 = [&](uint32_t data) {
    outFile.write(reinterpret_cast<const char *>(&data), sizeof(uint32_t));
  };
  write_uint32(super_feature_count_);
  for (int i = 0; i < super_feature_count_; i++) {
    write_uint64(index_[i].size());
    for (const auto &[k, v] : index_[i]) {
      write_uint64(k);
      write_uint32(v);
    }
  }
  return true;
}

bool SuperFeatureIndex::RecoverFromFile(const std::string &path) {
  std::ifstream inFile(path, std::ios::binary);
  if (!inFile) {
    LOG(FATAL) << "SuperFeatureIndex::RecoverFromFile: cannot open output file "
              << path;
    return false;
  }
  int super_feature_count = 0;
  inFile.read(reinterpret_cast<char *>(&super_feature_count), sizeof(int));
  if (super_feature_count_ != super_feature_count) {
    LOG(FATAL) << "super feature count changed after recover, abort";
    return false;
  }
  auto read_uint64 = [&]() -> uint64_t {
    uint64_t result = 0;
    inFile.read(reinterpret_cast<char *>(&result), sizeof(uint64_t));
    return result;
  };
  auto read_uint32 = [&]() -> uint32_t {
    uint32_t result = 0;
    inFile.read(reinterpret_cast<char *>(&result), sizeof(uint32_t));
    return result;
  };
  for (int i = 0; i < super_feature_count_; i++) {
    uint64_t mapSize = read_uint64();
    for (int j = 0; j < mapSize; j++) {
      auto feature = read_uint64();
      auto chunk_id = read_uint32();
      index_[i][feature] = chunk_id;
    }
  }
  return true;
}
} // namespace Delta
