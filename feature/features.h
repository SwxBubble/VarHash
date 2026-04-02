#pragma once
#include <string>
#include <memory>
#include <variant>
#include <vector>
namespace Delta {
constexpr int default_finesse_sf_cnt = 3;
// every super feature is grouped with 4 sub-features by default
constexpr int default_finesse_sf_subf = 4;

constexpr int default_odess_sf_cnt = 3;
constexpr int default_odess_sf_subf = 4;
constexpr uint64_t default_odess_mask = (1 << 7) - 1;
constexpr int default_varhash_bits = 128;
constexpr int default_varhash_segments = 8;
class Chunk;
using Feature = std::variant<std::vector<std::vector<uint64_t>>,
                             std::vector<uint64_t>
                             >;

class FeatureCalculator {
public:
  virtual Feature operator()(std::shared_ptr<Chunk> chunk) = 0;
};

class FinesseFeature : public FeatureCalculator {
public:
  FinesseFeature(const int sf_cnt = default_finesse_sf_cnt,
                 const int sf_subf = default_finesse_sf_subf)
      : sf_cnt_(sf_cnt), sf_subf_(sf_subf) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  // grouped super features count
  const int sf_cnt_;
  // how much sub feature does a one super feature contain
  const int sf_subf_;
};

class NTransformFeature : public FeatureCalculator {
public:
  NTransformFeature(const int sf_cnt = 3, const int sf_subf = 4)
      : sf_cnt_(sf_cnt), sf_subf_(sf_subf) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  // grouped super features count
  const int sf_cnt_;
  // how much sub feature does a one super feature contain
  const int sf_subf_;
};

class OdessFeature : public FeatureCalculator {
public:
  OdessFeature(const int sf_cnt = default_odess_sf_cnt,
               const int sf_subf = default_odess_sf_subf,
               const int mask = default_odess_mask)
      : sf_cnt_(sf_cnt), sf_subf_(sf_subf), mask_(mask) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  // grouped super features count
  const int sf_cnt_;
  // how much sub feature does a one super feature contain
  const int sf_subf_;

  const int mask_;
};

class OdessSubfeatures : public FeatureCalculator {
public:
  Feature operator()(std::shared_ptr<Chunk> chunk);
};

class PalantirFeature : public FeatureCalculator {
public:
  Feature operator()(std::shared_ptr<Chunk> chunk);
private:
  OdessSubfeatures get_sub_features_;
};

class VarHashFeature : public FeatureCalculator {
public:
  VarHashFeature(int hash_bits = default_varhash_bits,
                 int segment_count = default_varhash_segments,
                 std::string precomputed_hash_path = "")
      : hash_bits_(hash_bits), segment_count_(segment_count),
        precomputed_hash_path_(std::move(precomputed_hash_path)) {}

  Feature operator()(std::shared_ptr<Chunk> chunk) override;

private:
  int hash_bits_;
  int segment_count_;
  std::string precomputed_hash_path_;
};
} // namespace Delta
