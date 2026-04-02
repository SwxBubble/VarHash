#include "delta_compression.h"
#include "chunk/chunk.h"
#include "chunk/fast_cdc.h"
#include "chunk/rabin_cdc.h"
#include "config.h"
#include "encoder/xdelta.h"
#include "index/best_fit_index.h"
#include "index/hamming_index.h"
#include "index/palantir_index.h"
#include "index/super_feature_index.h"
#include "storage/storage.h"
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
namespace Delta {
namespace {
std::string GetConfigStringWithFallback(
    const std::shared_ptr<cpptoml::table> &config, const std::string &primary,
    const std::string &fallback = "") {
  auto value = config->get_as<std::string>(primary);
  if (value.has_value()) {
    return *value;
  }
  if (!fallback.empty()) {
    auto fallback_value = config->get_as<std::string>(fallback);
    if (fallback_value.has_value()) {
      return *fallback_value;
    }
  }
  return "";
}
} // namespace

std::optional<chunk_id>
DeltaCompression::SelectBestBaseChunk(const std::shared_ptr<Chunk> &chunk,
                                      const Feature &feature) {
  auto candidate_ids = index_->GetBaseChunkIDs(feature, top_k_candidates_);
  if (candidate_ids.empty()) {
    return std::nullopt;
  }
  size_t best_delta_size = chunk->len();
  std::optional<chunk_id> best_base_id = std::nullopt;
  for (auto candidate_id : candidate_ids) {
    auto delta_size = storage_->GetDeltaEncodedSize(chunk, candidate_id);
    if (delta_size < best_delta_size) {
      best_delta_size = delta_size;
      best_base_id = candidate_id;
    }
  }
  if (!best_base_id.has_value()) {
    return std::nullopt;
  }
  auto gain_bytes = chunk->len() - best_delta_size;
  if (gain_bytes < min_gain_bytes_) {
    return std::nullopt;
  }
  return best_base_id;
}

void DeltaCompression::AddFile(const std::string &file_name) {
  FileMeta file_meta;
  file_meta.file_name = file_name;
  file_meta.start_chunk_id = -1;
  this->chunker_->ReinitWithFile(file_name);
  while (true) {
    auto chunk = chunker_->GetNextChunk();
    if (nullptr == chunk)
      break;
    if (-1 == file_meta.start_chunk_id)
      file_meta.start_chunk_id = chunk->id();
    uint32_t dedup_base_id = dedup_->ProcessChunk(chunk);
    total_size_origin_ += chunk->len();
    // duplicate chunk
    if (dedup_base_id != chunk->id()) {
      storage_->WriteDuplicateChunk(chunk, dedup_base_id);
      duplicate_chunk_count_++;
      continue;
    }

    auto write_base_chunk = [this](const std::shared_ptr<Chunk> &chunk) {
      storage_->WriteBaseChunk(chunk);
      base_chunk_count_++;
      total_size_compressed_ += chunk->len();
    };

    auto write_delta_chunk = [this](const std::shared_ptr<Chunk> &chunk,
                                    const std::shared_ptr<Chunk> &delta_chunk,
                                    const uint32_t base_chunk_id) {
      chunk_size_before_delta_ += chunk->len();
      storage_->WriteDeltaChunk(delta_chunk, base_chunk_id);
      delta_chunk_count_++;
      chunk_size_after_delta_ += delta_chunk->len();
      total_size_compressed_ += delta_chunk->len();
    };

    auto feature = (*feature_)(chunk);
    auto base_chunk_id = SelectBestBaseChunk(chunk, feature);
    if (!base_chunk_id.has_value()) {
      index_->AddFeature(feature, chunk->id());
      write_base_chunk(chunk);
      continue;
    }

    auto delta_chunk =
        storage_->GetDeltaEncodedChunk(chunk, base_chunk_id.value());
    write_delta_chunk(chunk, delta_chunk, base_chunk_id.value());
    file_meta.end_chunk_id = chunk->id();
  }
  file_meta_writer_.Write(file_meta);
}

DeltaCompression::~DeltaCompression() {
  auto print_ratio = [](size_t a, size_t b) {
    double ratio = (double)a / (double)b;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "(" << ratio * 100 << "%)" << std::endl;
    std::cout << std::defaultfloat;
  };
  uint32_t chunk_count =
      base_chunk_count_ + delta_chunk_count_ + duplicate_chunk_count_;
  std::cout << "Total chunk count: " << chunk_count << std::endl;
  std::cout << "Base chunk count: " << base_chunk_count_;
  print_ratio(base_chunk_count_, chunk_count);
  std::cout << "Delta chunk count: " << delta_chunk_count_;
  print_ratio(delta_chunk_count_, chunk_count);
  std::cout << "Duplicate chunk count: " << duplicate_chunk_count_;
  print_ratio(duplicate_chunk_count_, chunk_count);
  std::cout << "DCR (Delta Compression Ratio): ";
  print_ratio(total_size_origin_, total_size_compressed_);
  std::cout << "before " << total_size_origin_
            << " after: " << total_size_compressed_ << std::endl;
  std::cout << "DCE (Delta Compression Efficiency): ";
  print_ratio(chunk_size_after_delta_, chunk_size_before_delta_);
}

#define declare_feature_type(NAME, FEATURE, INDEX)                             \
  {                                                                            \
#NAME, \
[]() -> FeatureIndex { \
  return {std::make_unique<FEATURE>(), \
          std::make_unique<INDEX>()}; \
}                                                                       \
  }

DeltaCompression::DeltaCompression() {
  auto config = Config::Instance().get();
  auto index_path = GetConfigStringWithFallback(config, "index_path",
                                                "feature_index_path");
  auto chunk_data_path = *config->get_as<std::string>("chunk_data_path");
  auto chunk_meta_path = *config->get_as<std::string>("chunk_meta_path");
  auto file_meta_path = *config->get_as<std::string>("file_meta_path");
  auto dedup_index_path = *config->get_as<std::string>("dedup_index_path");

  auto chunker = config->get_table("chunker");
  auto chunker_type = *chunker->get_as<std::string>("type");
  if (chunker_type == "rabin-cdc" || chunker_type == "fast-cdc") {
    auto min_chunk_size = *chunker->get_as<int64_t>("min_chunk_size");
    auto max_chunk_size = *chunker->get_as<int64_t>("max_chunk_size");
    auto stop_mask = *chunker->get_as<int64_t>("stop_mask");
    if (chunker_type == "rabin-cdc") {
      this->chunker_ =
          std::make_unique<RabinCDC>(min_chunk_size, max_chunk_size, stop_mask);
      LOG(INFO) << "Add RabinCDC chunker, min_chunk_size=" << min_chunk_size
                << " max_chunk_size=" << max_chunk_size
                << " stop_mask=" << stop_mask;
    } else if (chunker_type == "fast-cdc") {
      this->chunker_ =
          std::make_unique<FastCDC>(min_chunk_size, max_chunk_size, stop_mask);
      LOG(INFO) << "Add FastCDC chunker, min_chunk_size=" << min_chunk_size
                << " max_chunk_size=" << max_chunk_size
                << " stop_mask=" << stop_mask;
    }
  } else {
    LOG(FATAL) << "Unknown chunker type " << chunker_type;
  }

  auto feature = config->get_table("feature");
  auto feature_type = *feature->get_as<std::string>("type");
  top_k_candidates_ =
      static_cast<size_t>(feature->get_as<int64_t>("top_k_candidates")
                              .value_or<int64_t>(1));
  using FeatureIndex =
      std::pair<std::unique_ptr<FeatureCalculator>, std::unique_ptr<Index>>;
  std::unordered_map<std::string, std::function<FeatureIndex()>>
      feature_index_map = {
          declare_feature_type(finesse, FinesseFeature, SuperFeatureIndex),
          declare_feature_type(odess, OdessFeature, SuperFeatureIndex),
          declare_feature_type(n-transform, NTransformFeature,
                               SuperFeatureIndex),
          declare_feature_type(palantir, PalantirFeature, PalantirIndex),
          declare_feature_type(bestfit, OdessSubfeatures, BestFitIndex)};

  if (feature_type == "varhash") {
    auto hash_bits =
        feature->get_as<int64_t>("hash_bits").value_or(default_varhash_bits);
    auto segment_count = feature->get_as<int64_t>("segment_count")
                             .value_or(default_varhash_segments);
    auto precomputed_hash_path =
        feature->get_as<std::string>("precomputed_hash_path")
            .value_or(std::string(""));
    auto max_hamming_distance =
        feature->get_as<int64_t>("max_hamming_distance")
            .value_or(std::max<int64_t>(8, hash_bits / 8));
    this->feature_ = std::make_unique<VarHashFeature>(
        static_cast<int>(hash_bits), static_cast<int>(segment_count),
        precomputed_hash_path);
    this->index_ = std::make_unique<HammingIndex>(
        static_cast<uint32_t>(max_hamming_distance));
    LOG(INFO) << "Add VarHash feature, hash_bits=" << hash_bits
              << " segment_count=" << segment_count
              << " max_hamming_distance=" << max_hamming_distance;
  } else if (!feature_index_map.count(feature_type)) {
    LOG(FATAL) << "Unknown feature type " << feature_type;
  } else {
    auto [feature_ptr, index_ptr] = feature_index_map[feature_type]();
    this->feature_ = std::move(feature_ptr);
    this->index_ = std::move(index_ptr);
  }

  this->dedup_ = std::make_unique<Dedup>(dedup_index_path);

  auto storage = config->get_table("storage");
  auto encoder_name = *storage->get_as<std::string>("encoder");
  auto cache_size = *storage->get_as<int64_t>("cache_size");
  min_gain_bytes_ = static_cast<size_t>(
      storage->get_as<int64_t>("min_gain_bytes").value_or<int64_t>(1));
  std::unique_ptr<Encoder> encoder;
  if (encoder_name == "xdelta") {
    encoder = std::make_unique<XDelta>();
  } else {
    LOG(FATAL) << "Unknown encoder type " << encoder_name;
  }
  this->storage_ = std::make_unique<Storage>(
      chunk_data_path, chunk_meta_path, std::move(encoder), true, cache_size);
  this->file_meta_writer_.Init(file_meta_path);
}
} // namespace Delta
