// DataLoader.cpp
#include "DataLoader.h"

DataLoader::DataLoader(Dataset* dataset, size_t batch_size, bool shuffle, bool drop_last)
    : dataset_(dataset), batch_size_(batch_size),
    shuffle_(shuffle), drop_last_(drop_last), current_index_(0) {
    indices_.resize(dataset->size());
    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_) std::shuffle(indices_.begin(), indices_.end(), rng_);
}

bool DataLoader::has_next() const {
    return current_index_ < indices_.size();
}

DataLoader::Batch DataLoader::next() {
    size_t end = current_index_ + batch_size_;
    if (end > indices_.size()) {
        if (drop_last_) {
            current_index_ = indices_.size();
            return {};
        }
        end = indices_.size();
    }

    Batch batch;
    for (size_t i = current_index_; i < end; ++i) {
        batch.push_back(dataset_->get_item(indices_[i]));
    }

    current_index_ = end;
    return batch;
}

void DataLoader::reset() {
    current_index_ = 0;
    if (shuffle_) std::shuffle(indices_.begin(), indices_.end(), rng_);
}
