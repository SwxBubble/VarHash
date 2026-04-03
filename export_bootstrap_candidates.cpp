#include "3party/cpptoml.h"
#include "chunk/fast_cdc.h"
#include "chunk/rabin_cdc.h"
#include "config.h"
#include "dedup/dedup.h"
#include "feature/features.h"
#include "index/best_fit_index.h"
#include "index/hamming_index.h"
#include "index/palantir_index.h"
#include "index/super_feature_index.h"
#include "utils/sha1.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

DEFINE_string(config, "odess_export.toml", "path to config file");

namespace Delta {
namespace {

using FeatureIndex =
    std::pair<std::unique_ptr<FeatureCalculator>, std::unique_ptr<Index>>;

std::string StripArchiveSuffix(const std::string &name) {
  const std::vector<std::string> suffixes = {
      ".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".tbz2", ".txz", ".tar"};
  for (const auto &suffix : suffixes) {
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return name.substr(0, name.size() - suffix.size());
    }
  }
  auto pos = name.find_last_of('.');
  return pos == std::string::npos ? name : name.substr(0, pos);
}

std::vector<std::string> SplitVersionTokens(const std::string &version) {
  std::vector<std::string> tokens;
  std::string current;
  bool digit_mode = false;
  bool initialized = false;
  for (unsigned char ch : version) {
    bool is_digit = std::isdigit(ch) != 0;
    if (!initialized) {
      digit_mode = is_digit;
      initialized = true;
    }
    if (initialized && is_digit != digit_mode && !current.empty()) {
      tokens.push_back(current);
      current.clear();
      digit_mode = is_digit;
    }
    if (std::isalnum(ch) || ch == '.') {
      current.push_back(static_cast<char>(std::tolower(ch)));
    } else if (!current.empty()) {
      tokens.push_back(current);
      current.clear();
      initialized = false;
    }
  }
  if (!current.empty()) {
    tokens.push_back(current);
  }
  return tokens;
}

bool VersionLess(const std::string &lhs, const std::string &rhs) {
  const auto lhs_tokens = SplitVersionTokens(lhs);
  const auto rhs_tokens = SplitVersionTokens(rhs);
  const size_t limit = std::min(lhs_tokens.size(), rhs_tokens.size());
  for (size_t i = 0; i < limit; ++i) {
    const auto &a = lhs_tokens[i];
    const auto &b = rhs_tokens[i];
    const bool a_num = !a.empty() &&
                       std::all_of(a.begin(), a.end(), [](unsigned char ch) {
                         return std::isdigit(ch) != 0;
                       });
    const bool b_num = !b.empty() &&
                       std::all_of(b.begin(), b.end(), [](unsigned char ch) {
                         return std::isdigit(ch) != 0;
                       });
    if (a_num && b_num) {
      long long av = std::stoll(a);
      long long bv = std::stoll(b);
      if (av != bv) {
        return av < bv;
      }
    } else if (a != b) {
      return a < b;
    }
  }
  return lhs_tokens.size() < rhs_tokens.size();
}

std::string DigestToHex(const SHA1_digest &digest) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (int i = 0; i < 20; ++i) {
    oss << std::setw(2) << static_cast<int>(digest.d_[i]);
  }
  return oss.str();
}

struct ChunkInfo {
  std::string sha1;
  std::string project;
  std::string version;
  int version_order = 0;
  uint64_t chunk_offset = 0;
};

struct InputFile {
  std::filesystem::path relative_path;
  std::string project;
  std::string version;
  int version_order = 0;
};

std::vector<InputFile> CollectInputFiles(const std::filesystem::path &root) {
  std::unordered_map<std::string, std::vector<std::filesystem::path>> grouped;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    grouped[entry.path().parent_path().filename().string()].push_back(
        entry.path().lexically_relative(root));
  }

  std::vector<InputFile> results;
  for (auto &[project, files] : grouped) {
    std::sort(files.begin(), files.end(),
              [](const auto &lhs, const auto &rhs) {
                return VersionLess(StripArchiveSuffix(lhs.filename().string()),
                                   StripArchiveSuffix(rhs.filename().string()));
              });
    for (size_t i = 0; i < files.size(); ++i) {
      results.push_back({files[i], project,
                         StripArchiveSuffix(files[i].filename().string()),
                         static_cast<int>(i)});
    }
  }
  std::sort(results.begin(), results.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.project != rhs.project) {
                return lhs.project < rhs.project;
              }
              return lhs.version_order < rhs.version_order;
            });
  return results;
}

FeatureIndex CreateFeatureAndIndex(const std::shared_ptr<cpptoml::table> &config) {
  auto feature = config->get_table("feature");
  auto feature_type = *feature->get_as<std::string>("type");
  std::unordered_map<std::string, std::function<FeatureIndex()>> feature_index_map = {
      {"finesse", []() -> FeatureIndex {
         return {std::make_unique<FinesseFeature>(),
                 std::make_unique<SuperFeatureIndex>()};
       }},
      {"odess", []() -> FeatureIndex {
         return {std::make_unique<OdessFeature>(),
                 std::make_unique<SuperFeatureIndex>()};
       }},
      {"n-transform", []() -> FeatureIndex {
         return {std::make_unique<NTransformFeature>(),
                 std::make_unique<SuperFeatureIndex>()};
       }},
      {"palantir", []() -> FeatureIndex {
         return {std::make_unique<PalantirFeature>(),
                 std::make_unique<PalantirIndex>()};
       }},
      {"bestfit", []() -> FeatureIndex {
         return {std::make_unique<OdessSubfeatures>(),
                 std::make_unique<BestFitIndex>()};
       }},
      {"varhash", [feature]() -> FeatureIndex {
         auto hash_bits =
             feature->get_as<int64_t>("hash_bits").value_or(default_varhash_bits);
         auto segment_count =
             feature->get_as<int64_t>("segment_count").value_or(default_varhash_segments);
         auto precomputed_hash_path =
             feature->get_as<std::string>("precomputed_hash_path")
                 .value_or(std::string(""));
         auto max_hamming_distance =
             feature->get_as<int64_t>("max_hamming_distance")
                 .value_or(std::max<int64_t>(8, hash_bits / 8));
         return {std::make_unique<VarHashFeature>(static_cast<int>(hash_bits),
                                                  static_cast<int>(segment_count),
                                                  precomputed_hash_path),
                 std::make_unique<HammingIndex>(
                     static_cast<uint32_t>(max_hamming_distance))};
       }},
  };
  if (!feature_index_map.count(feature_type)) {
    LOG(FATAL) << "Unknown feature type " << feature_type;
  }
  return feature_index_map[feature_type]();
}

std::unique_ptr<Chunker> CreateChunker(const std::shared_ptr<cpptoml::table> &config) {
  auto chunker = config->get_table("chunker");
  auto chunker_type = *chunker->get_as<std::string>("type");
  auto min_chunk_size = *chunker->get_as<int64_t>("min_chunk_size");
  auto max_chunk_size = *chunker->get_as<int64_t>("max_chunk_size");
  auto stop_mask = *chunker->get_as<int64_t>("stop_mask");
  if (chunker_type == "fast-cdc") {
    return std::make_unique<FastCDC>(min_chunk_size, max_chunk_size, stop_mask);
  }
  if (chunker_type == "rabin-cdc") {
    return std::make_unique<RabinCDC>(min_chunk_size, max_chunk_size, stop_mask);
  }
  LOG(FATAL) << "Unknown chunker type " << chunker_type;
  return nullptr;
}

std::string JsonEscape(const std::string &value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char ch : value) {
    switch (ch) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out.push_back(ch);
      break;
    }
  }
  return out;
}

} // namespace

class BootstrapCandidateExporter {
public:
  BootstrapCandidateExporter() {
    auto config = Config::Instance().get();
    top_k_candidates_ = static_cast<size_t>(
        config->get_table("feature")
            ->get_as<int64_t>("top_k_candidates")
            .value_or<int64_t>(8));
    output_path_ =
        *config->get_as<std::string>("output_path");
    task_data_dir_ =
        *config->get_as<std::string>("task_data_dir");
    pair_type_ = *config->get_table("feature")->get_as<std::string>("type") +
                 std::string("_topk");
    chunker_ = CreateChunker(config);
    auto [feature_ptr, index_ptr] = CreateFeatureAndIndex(config);
    feature_ = std::move(feature_ptr);
    index_ = std::move(index_ptr);
    dedup_ = std::make_unique<Dedup>("");
  }

  void Run() {
    std::ofstream out(output_path_);
    if (!out) {
      LOG(FATAL) << "Cannot open output file " << output_path_;
    }
    const auto files = CollectInputFiles(task_data_dir_);
    if (files.empty()) {
      LOG(FATAL) << "No input files found under " << task_data_dir_.string();
    }
    size_t exported_pairs = 0;
    for (const auto &file : files) {
      const auto input_path = (task_data_dir_ / file.relative_path).string();
      if (!chunker_->ReinitWithFile(input_path)) {
        LOG(WARNING) << "Skip unreadable file " << input_path;
        continue;
      }
      LOG(INFO) << "Export candidates for " << input_path;
      uint64_t chunk_offset = 0;
      while (true) {
        auto chunk = chunker_->GetNextChunk();
        if (!chunk) {
          break;
        }
        const auto digest = sha1_hash(chunk->buf(), chunk->len());
        chunk_infos_[chunk->id()] = {
            DigestToHex(digest), file.project, file.version,
            file.version_order, chunk_offset};

        const auto dedup_base_id = dedup_->ProcessChunk(chunk);
        if (dedup_base_id != chunk->id()) {
          chunk_offset += static_cast<uint64_t>(chunk->len());
          continue;
        }

        const auto feature = (*feature_)(chunk);
        const auto candidate_ids = index_->GetBaseChunkIDs(feature, top_k_candidates_);
        for (size_t rank = 0; rank < candidate_ids.size(); ++rank) {
          const auto ref_id = candidate_ids[rank];
          if (!chunk_infos_.count(ref_id)) {
            continue;
          }
          const auto &query_info = chunk_infos_.at(chunk->id());
          const auto &ref_info = chunk_infos_.at(ref_id);
          out << "{\"query_sha1\":\"" << JsonEscape(query_info.sha1)
              << "\",\"ref_sha1\":\"" << JsonEscape(ref_info.sha1)
              << "\",\"pair_type\":\"" << JsonEscape(pair_type_)
              << "\",\"query_chunk_id\":" << chunk->id()
              << ",\"ref_chunk_id\":" << ref_id
              << ",\"project\":\"" << JsonEscape(file.project)
              << "\",\"version\":\"" << JsonEscape(file.version)
              << "\",\"version_order\":" << file.version_order
              << ",\"query_offset\":" << query_info.chunk_offset
              << ",\"ref_offset\":" << ref_info.chunk_offset
              << ",\"rank\":" << (rank + 1) << "}\n";
          exported_pairs++;
        }
        index_->AddFeature(feature, chunk->id());
        chunk_offset += static_cast<uint64_t>(chunk->len());
      }
    }
    LOG(INFO) << "Exported " << exported_pairs << " candidate pairs to "
              << output_path_;
  }

private:
  std::string output_path_;
  std::filesystem::path task_data_dir_;
  std::string pair_type_;
  size_t top_k_candidates_ = 8;
  std::unique_ptr<Chunker> chunker_;
  std::unique_ptr<FeatureCalculator> feature_;
  std::unique_ptr<Index> index_;
  std::unique_ptr<Dedup> dedup_;
  std::unordered_map<chunk_id, ChunkInfo> chunk_infos_;
};

} // namespace Delta

int main(int argc, char *argv[]) {
  FLAGS_stderrthreshold = google::INFO;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "using config file " << FLAGS_config;
  Delta::Config::Instance().Init(FLAGS_config);
  Delta::BootstrapCandidateExporter exporter;
  exporter.Run();
  return 0;
}
