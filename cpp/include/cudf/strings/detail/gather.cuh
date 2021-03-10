/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {

template <typename Iterator>
constexpr inline bool is_signed_iterator()
{
  return std::is_signed<typename std::iterator_traits<Iterator>::value_type>::value;
}

namespace strings {
namespace detail {

constexpr int strings_per_threadblock = 32;

/**
 * Binary search for at most strings_per_threadblock elements.
 *
 * Requires strings_per_threadblock to be an exponential of 2.
 * @param max_nelements Must be less than strings_per_threadblock.
 */
__forceinline__ __device__ cudf::size_type binary_search(cudf::size_type* offset,
                                                         cudf::size_type value,
                                                         cudf::size_type max_nelements)
{
  cudf::size_type idx = 0;
#pragma unroll
  for (cudf::size_type i = strings_per_threadblock / 2; i > 0; i /= 2) {
    if (idx + i < max_nelements && offset[idx + i] <= value) idx += i;
  }
  return idx;
}

template <bool NullifyOutOfBounds, typename MapIterator>
__global__ void gather_chars_fn(char* out_chars,
                                const char* in_chars,
                                const cudf::size_type* out_offsets,
                                const cudf::size_type* in_offsets,
                                MapIterator string_indices,
                                cudf::size_type total_out_strings,
                                cudf::size_type total_in_strings)
{
  if (in_chars == nullptr || in_offsets == nullptr) return;

  __shared__ cudf::size_type out_offsets_threadblock[strings_per_threadblock + 1];
  __shared__ cudf::size_type in_offsets_threadblock[strings_per_threadblock];

  // Current thread block will process output strings in range [begin_out_string_idx,
  // end_out_string_idx)
  cudf::size_type begin_out_string_idx = blockIdx.x * strings_per_threadblock;

  // Collectively load offsets of strings processed by the current thread block
  for (cudf::size_type idx = threadIdx.x; idx <= strings_per_threadblock; idx += blockDim.x) {
    cudf::size_type out_string_idx = idx + begin_out_string_idx;
    if (out_string_idx > total_out_strings) break;
    out_offsets_threadblock[idx] = out_offsets[out_string_idx];
  }

  for (cudf::size_type idx = threadIdx.x; idx < strings_per_threadblock; idx += blockDim.x) {
    cudf::size_type out_string_idx = idx + begin_out_string_idx;
    if (out_string_idx >= total_out_strings) break;
    cudf::size_type in_string_idx = string_indices[out_string_idx];
    if (NullifyOutOfBounds) {
      if (is_signed_iterator<MapIterator>()
            ? ((in_string_idx < 0) || (in_string_idx >= total_in_strings))
            : (in_string_idx >= total_in_strings))
        continue;
    }
    in_offsets_threadblock[idx] = in_offsets[in_string_idx];
  }
  __syncthreads();

  cudf::size_type strings_current_threadblock =
    min(strings_per_threadblock, total_out_strings - begin_out_string_idx);

  for (int out_ibyte = threadIdx.x + out_offsets_threadblock[0];
       out_ibyte < out_offsets_threadblock[strings_current_threadblock];
       out_ibyte += blockDim.x) {
    // binary search for the string index corresponding to out_ibyte
    cudf::size_type string_idx =
      binary_search(out_offsets_threadblock, out_ibyte, strings_current_threadblock);

    // calculate which character to load within the string
    cudf::size_type icharacter = out_ibyte - out_offsets_threadblock[string_idx];

    out_chars[out_ibyte] = in_chars[in_offsets_threadblock[string_idx] + icharacter];
  }
}

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * Caller must update the validity mask in the output column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather<true>( s1, map.begin(), map.end() )
 * s2 is ["a", "c"]
 * ```
 *
 * @tparam NullifyOutOfBounds If true, indices outside the column's range are nullified.
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New strings column containing the gathered strings.
 */
template <bool NullifyOutOfBounds, typename MapIterator>
std::unique_ptr<cudf::column> gather(
  strings_column_view const& strings,
  MapIterator begin,
  MapIterator end,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const output_count  = std::distance(begin, end);
  auto const strings_count = strings.size();
  if (output_count == 0) return make_empty_strings_column(stream, mr);

  // allocate offsets column and use memory to compute string size in each output row
  auto out_offsets_column = make_numeric_column(
    data_type{type_id::INT32}, output_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto const d_out_offsets = out_offsets_column->mutable_view().template data<int32_t>();
  auto const d_in_offsets =
    (strings_count > 0) ? strings.offsets().data<int32_t>() + strings.offset() : nullptr;
  thrust::transform(rmm::exec_policy(stream),
                    begin,
                    end,
                    d_out_offsets,
                    [d_in_offsets, strings_count] __device__(size_type in_idx) {
                      if (NullifyOutOfBounds && (in_idx < 0 || in_idx >= strings_count)) return 0;
                      return d_in_offsets[in_idx + 1] - d_in_offsets[in_idx];
                    });

  // check total size is not too large
  size_t const total_bytes = thrust::transform_reduce(
    rmm::exec_policy(stream),
    d_out_offsets,
    d_out_offsets + output_count,
    [] __device__(auto size) { return static_cast<size_t>(size); },
    size_t{0},
    thrust::plus<size_t>{});
  CUDF_EXPECTS(total_bytes < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "total size of output strings is too large for a cudf column");

  // In-place convert output sizes into offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_out_offsets, d_out_offsets + output_count + 1, d_out_offsets);

  // build chars column
  size_type const out_chars_bytes = static_cast<size_type>(total_bytes);
  auto out_chars_column  = create_chars_child_column(output_count, 0, out_chars_bytes, stream, mr);
  auto const d_out_chars = out_chars_column->mutable_view().template data<char>();

  // fill in chars
  auto const d_in_chars = (strings_count > 0) ? strings.chars().data<char>() : nullptr;

  int num_threadblocks = (output_count + strings_per_threadblock - 1) / strings_per_threadblock;
  gather_chars_fn<NullifyOutOfBounds, MapIterator><<<num_threadblocks, 128, 0, stream.value()>>>(
    d_out_chars, d_in_chars, d_out_offsets, d_in_offsets, begin, output_count, strings_count);

  return make_strings_column(output_count,
                             std::move(out_offsets_column),
                             std::move(out_chars_column),
                             0,
                             rmm::device_buffer{0, stream, mr},
                             stream,
                             mr);
}

/**
 * @brief Returns a new strings column using the specified indices to select
 * elements from the `strings` column.
 *
 * Caller must update the validity mask in the output column.
 *
 * ```
 * s1 = ["a", "b", "c", "d", "e", "f"]
 * map = [0, 2]
 * s2 = gather( s1, map.begin(), map.end(), true )
 * s2 is ["a", "c"]
 * ```
 *
 * @tparam MapIterator Iterator for retrieving integer indices of the column.
 *
 * @param strings Strings instance for this operation.
 * @param begin Start of index iterator.
 * @param end End of index iterator.
 * @param nullify_out_of_bounds If true, indices outside the column's range are nullified.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New strings column containing the gathered strings.
 */
template <typename MapIterator>
std::unique_ptr<cudf::column> gather(
  strings_column_view const& strings,
  MapIterator begin,
  MapIterator end,
  bool nullify_out_of_bounds,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (nullify_out_of_bounds) return gather<true>(strings, begin, end, stream, mr);
  return gather<false>(strings, begin, end, stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
