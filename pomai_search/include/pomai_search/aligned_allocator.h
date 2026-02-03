#pragma once

#include <cstdlib>
#include <memory>

namespace pomai_search {

template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;

  explicit AlignedAllocator(size_t alignment = 32) noexcept : alignment_(alignment) {}

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U>& other) noexcept : alignment_(other.alignment()) {}

  T* allocate(std::size_t n) {
    void* ptr = nullptr;
    size_t size = n * sizeof(T);
    if (posix_memalign(&ptr, alignment_, size) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) noexcept { free(p); }

  size_t alignment() const noexcept { return alignment_; }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U>;
  };

 private:
  size_t alignment_;
};

template <typename T, typename U>
bool operator==(const AlignedAllocator<T>& lhs, const AlignedAllocator<U>& rhs) {
  return lhs.alignment() == rhs.alignment();
}

template <typename T, typename U>
bool operator!=(const AlignedAllocator<T>& lhs, const AlignedAllocator<U>& rhs) {
  return !(lhs == rhs);
}

}  // namespace pomai_search
