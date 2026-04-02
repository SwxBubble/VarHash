#include "feature/features.h"
#include "chunk/chunk.h"
#include "utils/gear.h"
#include "utils/rabin.cpp"
#include "utils/sha1.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>
#include <unordered_map>
namespace Delta {
Feature FinesseFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int sub_chunk_length = chunk->len() / (sf_subf_ * sf_cnt_);
  uint8_t *content = chunk->buf();
  std::vector<uint64_t> sub_features(sf_cnt_ * sf_subf_, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  // calculate sub features.
  for (int i = 0; i < sub_features.size(); i++) {
    rabin_t rabin_ctx;
    rabin_init(&rabin_ctx);
    for (int j = 0; j < sub_chunk_length; j++) {
      rabin_append(&rabin_ctx, content[j]);
      sub_features[i] = std::max(rabin_ctx.digest, sub_features[i]);
    }
    content += sub_chunk_length;
  }

  // group the sub features into super features.
  for (int i = 0; i < sub_features.size(); i += sf_subf_) {
    std::sort(sub_features.begin() + i, sub_features.begin() + i + sf_subf_);
  }
  for (int i = 0; i < sf_cnt_; i++) {
    rabin_t rabin_ctx;
    rabin_init(&rabin_ctx);
    for (int j = 0; j < sf_subf_; j++) {
      auto sub_feature = sub_features[sf_subf_ * i + j];
      auto data_ptr = (uint8_t *)&sub_feature;
      for (int k = 0; k < 8; k++) {
        rabin_append(&rabin_ctx, data_ptr[k]);
      }
    }
    super_features[i] = rabin_ctx.digest;
  }
  return super_features;
}

static uint32_t M[] = {
    0x5b49898a, 0xe4f94e27, 0x95f658b2, 0x8f9c99fc, 0xeba8d4d8, 0xba2c8e92,
    0xa868aeb4, 0xd767df82, 0x843606a4, 0xc1e70129, 0x32d9d1b0, 0xeb91e53c,
};

static uint32_t A[] = {
    0xff4be8c,  0x6f485986, 0x12843ff,  0x5b47dc4d, 0x7faa9b8a, 0xd547b8ba,
    0xf9979921, 0x4f5400da, 0x725f79a9, 0x3c9321ac, 0x32716d,   0x3f5adf5d,
};

Feature NTransformFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int features_num = sf_cnt_ * sf_subf_;
  std::vector<uint32_t> sub_features(features_num, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint64_t finger_print = 0;
  // calculate sub features.
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
    for (int j = 0; j < features_num; j++) {
      const uint32_t transform = (M[j] * finger_print + A[j]);
      // we need to guarantee that when sub_features[i] is not inited,
      // always set its value
      if (sub_features[j] >= transform || 0 == sub_features[j])
        sub_features[j] = transform;
    }
  }

  // group sub features into super features.
  auto hash_buf = (const uint8_t *const)(sub_features.data());
  for (int i = 0; i < sf_cnt_; i++) {
    uint64_t hash_value = 0;
    auto this_hash_buf = hash_buf + i * sf_subf_ * sizeof(uint32_t);
    for (int j = 0; j < sf_subf_ * sizeof(uint32_t); j++) {
      hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
    }
    super_features[i] = hash_value;
  }
  return super_features;
}

Feature OdessFeature::operator()(std::shared_ptr<Chunk> chunk) {
  int features_num = sf_cnt_ * sf_subf_;
  std::vector<uint32_t> sub_features(features_num, 0);
  std::vector<uint64_t> super_features(sf_cnt_, 0);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint64_t finger_print = 0;
  // calculate sub features.
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
    if ((finger_print & mask_) == 0) {
      for (int j = 0; j < features_num; j++) {
        const uint32_t transform = (M[j] * finger_print + A[j]);
        // we need to guarantee that when sub_features[i] is not inited,
        // always set its value
        if (sub_features[j] >= transform || 0 == sub_features[j])
          sub_features[j] = transform;
      }
    }
  }

  // group sub features into super features.
  auto hash_buf = (const uint8_t *const)(sub_features.data());
  for (int i = 0; i < sf_cnt_; i++) {
    uint64_t hash_value = 0;
    auto this_hash_buf = hash_buf + i * sf_subf_ * sizeof(uint32_t);
    for (int j = 0; j < sf_subf_ * sizeof(uint32_t); j++) {
      hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
    }
    super_features[i] = hash_value;
  }
  return super_features;
}

Feature OdessSubfeatures::operator()(std::shared_ptr<Chunk> chunk) {
  int mask_ = default_odess_mask;
  int features_num = 12;
  std::vector<uint64_t> sub_features(features_num, 0);

  int chunk_length = chunk->len();
  uint8_t *content = chunk->buf();
  uint32_t finger_print = 0;
  // calculate sub features.
  for (int i = 0; i < chunk_length; i++) {
    finger_print = (finger_print << 1) + GEAR_TABLE[content[i]];
    if ((finger_print & mask_) == 0) {
      for (int j = 0; j < features_num; j++) {
        const uint64_t transform = (M[j] * finger_print + A[j]);
        // we need to guarantee that when sub_features[i] is not inited,
        // always set its value
        if (sub_features[j] >= transform || 0 == sub_features[j])
          sub_features[j] = transform;
      }
    }
  }

  return sub_features;
}

Feature PalantirFeature::operator()(std::shared_ptr<Chunk> chunk) {
  auto sub_features = std::get<std::vector<uint64_t>>(get_sub_features_(chunk));
  std::vector<std::vector<uint64_t>> results;

  auto group = [&](int sf_cnt, int sf_subf) -> std::vector<uint64_t> {
    std::vector<uint64_t> super_features(sf_cnt, 0);
    auto hash_buf = (const uint8_t *const)(sub_features.data());
    for (int i = 0; i < sf_cnt; i++) {
      uint64_t hash_value = 0;
      auto this_hash_buf = hash_buf + i * sf_subf * sizeof(uint64_t);
      for (int j = 4; j < sf_subf * sizeof(uint64_t); j++) {
        hash_value = (hash_value << 1) + GEAR_TABLE[this_hash_buf[j]];
      }
      super_features[i] = hash_value;
    }
    return super_features;
  };

  results.push_back(group(3, 4));
  results.push_back(group(4, 3));
  results.push_back(group(6, 2));
  return results;
}

namespace {
constexpr int kHistogramBins = 16;

uint64_t SplitMix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31U);
}

std::string DigestToHex(const SHA1_digest &digest) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < 20; ++i) {
    oss << std::setw(2) << static_cast<int>(digest.d_[i]);
  }
  return oss.str();
}

double ByteEntropy(const std::array<uint64_t, 256> &counts, int total) {
  if (total <= 0) {
    return 0.0;
  }
  double entropy = 0.0;
  for (uint64_t count : counts) {
    if (count == 0) {
      continue;
    }
    double p = static_cast<double>(count) / static_cast<double>(total);
    entropy -= p * std::log2(p);
  }
  return entropy / 8.0;
}

std::vector<double> BuildDescriptor(const std::shared_ptr<Chunk> &chunk,
                                    int segment_count) {
  const int length = chunk->len();
  auto *buf = chunk->buf();
  std::array<uint64_t, 256> raw_histogram{};
  std::array<double, kHistogramBins> coarse_histogram{};
  std::vector<double> descriptor;
  descriptor.reserve(kHistogramBins + segment_count * 4 + 4);

  uint64_t nonzero_count = 0;
  uint64_t transition_sum = 0;
  uint64_t transition_large_count = 0;
  uint64_t byte_sum = 0;
  for (int i = 0; i < length; ++i) {
    uint8_t value = buf[i];
    raw_histogram[value]++;
    coarse_histogram[value / (256 / kHistogramBins)] += 1.0;
    byte_sum += value;
    if (value != 0) {
      nonzero_count++;
    }
    if (i > 0) {
      auto diff = static_cast<uint32_t>(std::abs(buf[i] - buf[i - 1]));
      transition_sum += diff;
      if (diff >= 16) {
        transition_large_count++;
      }
    }
  }

  double inv_length = length > 0 ? 1.0 / static_cast<double>(length) : 0.0;
  for (double &bin : coarse_histogram) {
    bin *= inv_length;
    descriptor.push_back(bin);
  }

  int segment_len =
      std::max(1, (length + std::max(1, segment_count) - 1) / std::max(1, segment_count));
  for (int segment = 0; segment < std::max(1, segment_count); ++segment) {
    int start = segment * segment_len;
    if (start >= length) {
      descriptor.insert(descriptor.end(), {0.0, 0.0, 0.0, 0.0});
      continue;
    }
    int end = std::min(length, start + segment_len);
    double local_sum = 0.0;
    double local_square_sum = 0.0;
    double local_transition = 0.0;
    uint64_t rolling_min = std::numeric_limits<uint64_t>::max();
    uint64_t rolling = 0;
    int local_count = end - start;
    for (int i = start; i < end; ++i) {
      double value = static_cast<double>(buf[i]) / 255.0;
      local_sum += value;
      local_square_sum += value * value;
      rolling = (rolling << 1U) + GEAR_TABLE[buf[i]];
      rolling_min = std::min<uint64_t>(rolling_min, rolling);
      if (i > start) {
        local_transition += static_cast<double>(std::abs(buf[i] - buf[i - 1])) / 255.0;
      }
    }
    double mean = local_count > 0 ? local_sum / local_count : 0.0;
    double variance = local_count > 0 ? std::max(0.0, local_square_sum / local_count - mean * mean) : 0.0;
    double transition = local_count > 1 ? local_transition / (local_count - 1) : 0.0;
    double anchor = rolling_min == std::numeric_limits<uint64_t>::max()
                        ? 0.0
                        : static_cast<double>(rolling_min & 0xFFFFU) / 65535.0;
    descriptor.push_back(mean);
    descriptor.push_back(variance);
    descriptor.push_back(transition);
    descriptor.push_back(anchor);
  }

  descriptor.push_back(static_cast<double>(length) / 16384.0);
  descriptor.push_back(length > 0 ? static_cast<double>(byte_sum) / static_cast<double>(length) / 255.0 : 0.0);
  descriptor.push_back(length > 0 ? static_cast<double>(nonzero_count) / static_cast<double>(length) : 0.0);
  descriptor.push_back(length > 1 ? static_cast<double>(transition_sum) / static_cast<double>(length - 1) / 255.0 : 0.0);
  descriptor.push_back(length > 1 ? static_cast<double>(transition_large_count) / static_cast<double>(length - 1) : 0.0);
  descriptor.push_back(ByteEntropy(raw_histogram, length));
  return descriptor;
}

std::vector<uint64_t> DescriptorToHashWords(const std::vector<double> &descriptor,
                                            int hash_bits) {
  int word_count = (hash_bits + 63) / 64;
  std::vector<uint64_t> words(word_count, 0);
  for (int bit = 0; bit < hash_bits; ++bit) {
    double score = 0.0;
    for (size_t dim = 0; dim < descriptor.size(); ++dim) {
      uint64_t seed = SplitMix64((static_cast<uint64_t>(bit) << 32U) ^
                                 static_cast<uint64_t>(dim + 1));
      double weight = static_cast<double>((seed & 0xFFFFU)) / 32767.5 - 1.0;
      score += descriptor[dim] * weight;
    }
    uint64_t bias_seed = SplitMix64(static_cast<uint64_t>(bit) ^ 0xa5a5a5a5U);
    double bias = static_cast<double>((bias_seed & 0xFFU)) / 255.0 - 0.5;
    if (score + bias >= 0.0) {
      words[bit / 64] |= (1ULL << (bit % 64));
    }
  }
  return words;
}

std::unordered_map<std::string, std::vector<uint64_t>>
LoadPrecomputedHashes(const std::string &path) {
  std::unordered_map<std::string, std::vector<uint64_t>> result;
  if (path.empty()) {
    return result;
  }
  std::ifstream input(path);
  if (!input) {
    return result;
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    std::string digest_hex;
    iss >> digest_hex;
    if (digest_hex.empty()) {
      continue;
    }
    std::vector<uint64_t> words;
    std::string word_hex;
    while (iss >> word_hex) {
      std::stringstream converter;
      converter << std::hex << word_hex;
      uint64_t value = 0;
      converter >> value;
      words.push_back(value);
    }
    if (!words.empty()) {
      result.emplace(std::move(digest_hex), std::move(words));
    }
  }
  return result;
}
} // namespace

Feature VarHashFeature::operator()(std::shared_ptr<Chunk> chunk) {
  static std::unordered_map<std::string, std::vector<uint64_t>> precomputed_hashes =
      LoadPrecomputedHashes(precomputed_hash_path_);
  if (!precomputed_hash_path_.empty() && !precomputed_hashes.empty()) {
    auto digest = DigestToHex(sha1_hash(chunk->buf(), chunk->len()));
    auto iter = precomputed_hashes.find(digest);
    if (iter != precomputed_hashes.end()) {
      return iter->second;
    }
  }
  auto descriptor = BuildDescriptor(chunk, segment_count_);
  return DescriptorToHashWords(descriptor, hash_bits_);
}
} // namespace Delta
