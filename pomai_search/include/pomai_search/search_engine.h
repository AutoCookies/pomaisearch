#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/serialize/snapshot.h"
#include "pomai_search/status.h"
#include "pomai_search/types.h"

namespace pomai_search {

/**
 * @brief High-performance vector search engine with hybrid search capabilities.
 * 
 * SearchEngine provides vector similarity search with support for multiple index types
 * (Flat, HNSW, IVF-Flat, IVF-SQ8), metadata filtering, TTL expiration, and keyword search.
 * 
 * Features:
 * - Multiple index types optimized for different use cases
 * - Hybrid vector + keyword search
 * - Metadata filtering and TTL support
 * - Native binary serialization for fast startup
 * - Thread-safe operations
 * 
 * @example
 * ```cpp
 * SearchEngineConfig cfg;
 * cfg.dim = 128;
 * cfg.index_type = SearchEngineConfig::IndexType::Hnsw;
 * auto engine = SearchEngine::Open(cfg).value();
 * 
 * float vec[128] = {...};
 * engine->Upsert("doc1", VectorView{vec, 128}, {{"category", "tech"}});
 * auto results = engine->Search(VectorView{vec, 128});
 * ```
 */
class SearchEngine {
 public:
  /**
   * @brief Query options for search operations.
   */
  struct QueryOptions {
    int topk = 0;        ///< Number of results to return (0 = use default)
    Filter filter;       ///< Metadata and expiry filters
    QueryOptions() = default;
  };

  /**
   * @brief Hybrid search query combining vector and text search.
   */
  struct HybridQuery {
    std::optional<std::string> text_query;           ///< Optional keyword query
    std::optional<std::vector<float>> vector_query;  ///< Optional vector query
    float alpha = 0.5f;  ///< Weight for vector vs text (0=text only, 1=vector only)
  };

  /**
   * @brief Statistics about the search engine state.
   */
  struct Stats {
    uint64_t num_points = 0;   ///< Total number of indexed vectors
    uint64_t num_deleted = 0;  ///< Number of deleted vectors
  };

  /**
   * @brief Opens a new search engine instance with the given configuration.
   * 
   * @param cfg Configuration specifying dimensions, index type, and parameters
   * @return StatusOr containing the engine instance or an error status
   * 
   * @note The engine must be explicitly closed with Close() or destroyed
   */
  static StatusOr<std::unique_ptr<SearchEngine>> Open(const SearchEngineConfig& cfg);

  /**
   * @brief Inserts or updates a vector with associated metadata.
   * 
   * @param key Unique identifier for the vector
   * @param vec Vector data to index
   * @param meta Optional metadata key-value pairs
   * @param ttl Optional time-to-live duration
   * @param text Optional text content for hybrid search
   * @return Status indicating success or failure
   * 
   * @note If key exists, the vector and metadata are updated
   */
  Status Upsert(std::string_view key, VectorView vec, Metadata meta = {},
                std::optional<std::chrono::milliseconds> ttl = std::nullopt,
                std::optional<std::string> text = std::nullopt);

  /**
   * @brief Deletes a vector by key.
   * 
   * @param key Key of the vector to delete
   * @return Status indicating success or kNotFound if key doesn't exist
   */
  Status Delete(std::string_view key);

  /**
   * @brief Checks if a key exists and is not expired.
   * 
   * @param key Key to check
   * @return StatusOr containing true if exists, false otherwise, or error status
   */
  StatusOr<bool> Exists(std::string_view key) const;

  /**
   * @brief Retrieves the vector associated with a key.
   * 
   * @param key Key of the vector to retrieve
   * @return StatusOr containing the vector or kNotFound error
   */
  StatusOr<std::vector<float>> GetVector(std::string_view key) const;

  /**
   * @brief Searches for similar vectors.
   * 
   * @param q Query vector
   * @param opt Query options (topk, filters)
   * @return StatusOr containing ranked results or error status
   * 
   * @note Results are sorted by similarity score (highest first)
   */
  StatusOr<std::vector<ResultItem>> Search(VectorView q, QueryOptions opt);

  /// @brief Convenience overload using default QueryOptions
  StatusOr<std::vector<ResultItem>> Search(VectorView q) { return Search(q, QueryOptions{}); }

  /**
   * @brief Searches using an existing vector as the query.
   * 
   * @param key Key of the vector to use as query
   * @param opt Query options
   * @return StatusOr containing ranked results or error status
   */
  StatusOr<std::vector<ResultItem>> SearchByKey(std::string_view key, QueryOptions opt);

  /// @brief Convenience overload using default QueryOptions
  StatusOr<std::vector<ResultItem>> SearchByKey(std::string_view key) {
    return SearchByKey(key, QueryOptions{});
  }

  /**
   * @brief Performs hybrid vector + keyword search.
   * 
   * @param query Hybrid query with optional vector and text components
   * @param opt Query options
   * @return StatusOr containing ranked results combining both signals
   * 
   * @note Results are ranked by weighted combination: alpha * vector_score + (1-alpha) * text_score
   */
  StatusOr<std::vector<ResultItem>> SearchHybrid(const HybridQuery& query, QueryOptions opt);

  /// @brief Convenience overload using default QueryOptions
  StatusOr<std::vector<ResultItem>> SearchHybrid(const HybridQuery& query) {
    return SearchHybrid(query, QueryOptions{});
  }

  /**
   * @brief Searches with detailed query explanation.
   * 
   * @param q Query vector
   * @param opt Query options
   * @param policy Query policy for explain information
   * @return StatusOr containing results with explain metadata
   */
  StatusOr<SearchResponse> SearchWithExplain(VectorView q, QueryOptions opt, QueryPolicy policy);

  /**
   * @brief Hybrid search with detailed query explanation.
   * 
   * @param query Hybrid query
   * @param opt Query options
   * @param policy Query policy for explain information
   * @return StatusOr containing results with explain metadata
   */
  StatusOr<SearchResponse> SearchHybridWithExplain(const HybridQuery& query, QueryOptions opt,
                                                   QueryPolicy policy);

  /**
   * @brief Returns current engine statistics.
   * 
   * @return Stats structure with point counts
   */
  Stats GetStats() const;

  /**
   * @brief Returns the configured vector dimension.
   */
  int Dim() const;

  /**
   * @brief Exports all non-expired records for iteration or snapshotting.
   *
   * @param filter Optional metadata filter to apply.
   * @return StatusOr containing exported records or an error status.
   */
  StatusOr<std::vector<SnapshotRecord>> ExportRecords(const Filter& filter) const;

  /**
   * @brief Returns metrics in JSON format.
   * 
   * @return JSON string with query latencies and other metrics
   */
  std::string MetricsJson() const;

  /**
   * @brief Closes the engine and releases resources.
   * 
   * @return Status indicating success or failure
   * 
   * @note The engine cannot be used after calling Close()
   */
  Status Close();

  /**
   * @brief Destructor automatically closes the engine.
   */
  ~SearchEngine();

 private:
  SearchEngine();
  Status Initialize(const SearchEngineConfig& cfg);

  friend class SnapshotWriter;
  friend class SnapshotReader;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pomai_search
