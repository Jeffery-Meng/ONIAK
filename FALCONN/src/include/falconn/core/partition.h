#ifndef __PARTITION_H__
#define __PARTITION_H__

#include <memory>
#include <utility>
#include <vector>
#include <fstream>
#include "hash_table_helpers.h"
#include "composite_hash_table.h"
#include "partition_metric.h"

namespace falconn {
namespace core {


template <typename KeyType, typename ValueType, typename PartitionMetricType, typename InnerHashTable>
class StaticPartition
    : public BasicCompositeHashTable<KeyType, ValueType, InnerHashTable> {
 public:
  StaticPartition(int_fast32_t l, 
                  typename InnerHashTable::Factory* factory,
                  PartitionMetricType lower, PartitionMetricType upper)
      : BasicCompositeHashTable<KeyType, ValueType, InnerHashTable>(l, factory),
      _lower(lower), _upper(upper)
       {}

  void add_entries_for_table(const std::vector<KeyType>& keys,
                             int_fast32_t table) {
    if (table < 0 || table >= this->l_) {
      throw CompositeHashTableError("Table index incorrect.");
    }

    this->tables_[table]->add_entries(keys);
  }

  void add_entries_in_keys(const std::vector<KeyType>& hash_keys, const std::vector<ValueType>& keys,
                             int_fast32_t table) {
    if (table < 0 || table >= this->l_) {
      throw CompositeHashTableError("Table index incorrect.");
    }

    this->tables_[table]->add_entries_in_keys(hash_keys, keys);
  }

  void dump_table_to_stream(std::ofstream& fout, int_fast32_t table){
    if (table < 0 || table >= this->l_) {
      throw CompositeHashTableError("Table index incorrect.");
    }

    this->tables_[table]->dump_entries_to_stream(fout);
  }

  void add_entries_from_stream(std::ifstream& fin, int_fast32_t table){
    if (table < 0 || table >= this->l_) {
      throw CompositeHashTableError("Table index incorrect.");
    }

    this->tables_[table]->add_entries_from_stream(fin);
  }

  PartitionMetricType get_lower() const {
    return _lower;
  }

  PartitionMetricType get_upper() const {
    return _upper;
  }

  private:
    PartitionMetricType _lower, _upper;
};

}  // namespace core
}  // namespace falconn

#endif
