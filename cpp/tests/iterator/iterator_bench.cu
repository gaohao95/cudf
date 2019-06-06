/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <chrono>
#include <random>
#include <tuple>

#include <cuda_profiler_api.h>
#include <utilities/error_utils.hpp>
#include <tests/utilities/column_wrapper.cuh>

#include <iterator/iterator.cuh>    // include iterator header
#include <utilities/device_operators.cuh>

// for reduction tests
#include <cub/device/device_reduce.cuh>
#include <thrust/device_vector.h>
#include <reduction.hpp>


template <typename T>
T random_int(T min, T max)
{
  static unsigned seed = 13377331;
  static std::mt19937 engine{seed};
  static std::uniform_int_distribution<T> uniform{min, max};

  return uniform(engine);
}

gdf_dtype type_from_name(const std::string &name)
{
  if      (name == "a") return GDF_INT8;
  else if (name == "s") return GDF_INT16;
  else if (name == "i") return GDF_INT32;
  else if (name == "l") return GDF_INT64;
  else if (name == "f") return GDF_FLOAT32;
  else if (name == "d") return GDF_FLOAT64;
  else return N_GDF_TYPES;
}
const char* name_from_type(gdf_dtype type)
{
  switch (type) {
    case GDF_INT8:    return "GDF_INT8";
    case GDF_INT16:   return "GDF_INT16";
    case GDF_INT32:   return "GDF_INT32";
    case GDF_INT64:   return "GDF_INT64";
    case GDF_FLOAT32: return "GDF_FLOAT32";
    case GDF_FLOAT64: return "GDF_FLOAT64";
    default:          return "GDF_INVALID";
  }
}

// -----------------------------------------------------------------------------

class BenchMarkTimer
{
public:
  BenchMarkTimer(int iters_) : iters(iters_)
  {
    start();
  };
  ~BenchMarkTimer(){end();};

protected:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
  int iters;

  void start()
  {
    cudaProfilerStart();
    start_point = std::chrono::high_resolution_clock::now();
  };

  void end()
  {
    cudaDeviceSynchronize();
    auto end_point = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_point-start_point;
    cudaProfilerStop();
    std::cout << diff.count() / iters  << std::endl << std::flush;
  };

};

// -----------------------------------------------------------------------------
template <typename InputIterator, typename OutputIterator, typename T>
inline void reduce_by_cub_storage(
  void *d_temp_storage, size_t& temp_storage_bytes,
  InputIterator d_in, OutputIterator result, int num_items, T init)
{
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, result, num_items,
      cudf::DeviceSum{}, init);
}

template <typename InputIterator, typename OutputIterator, typename T>
inline auto reduce_by_cub(OutputIterator result, InputIterator d_in, int num_items, T init)
{
  void     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;

  reduce_by_cub_storage(d_temp_storage, temp_storage_bytes, d_in, result, num_items, init);

  // Allocate temporary storage
  RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));

  // Run reduction
  reduce_by_cub_storage(d_temp_storage, temp_storage_bytes, d_in, result, num_items, init);

  // Free temporary storage
  RMM_TRY(RMM_FREE(d_temp_storage, 0));

  return temp_storage_bytes;
}
// ------------------------

template <typename T>
void raw_stream_bench_cub(cudf::test::column_wrapper<T>& col, rmm::device_vector<T>& result, int iters, bool no_new_allocate=false)
{
  std::cout << "raw strem cub: " << "\t\t";

  T init{0};
  auto begin = static_cast<T*>(col.get()->data);
  int num_items = col.size();

  if( no_new_allocate ){
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    auto bench = [&](){ reduce_by_cub_storage(d_temp_storage, temp_storage_bytes, begin, result.begin(), num_items, init);};

    bench();
    // Allocate temporary storage
    RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));
    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);

    // Free temporary storage
    RMM_TRY(RMM_FREE(d_temp_storage, 0));
  }else{
    auto bench = [&](){ reduce_by_cub(result.begin(), begin, num_items, init);};

    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);
  }
};

template <typename T, bool has_null>
void iterator_bench_cub(cudf::test::column_wrapper<T>& col, rmm::device_vector<T>& result, int iters, bool no_new_allocate=false)
{

  std::cout << "iterator cub " << ( (has_null) ? "<true>: " : "<false>: " ) << "\t";

  T init{0};
  auto begin = cudf::make_iterator<has_null, T>(col, init);
  int num_items = col.size();

  if( no_new_allocate ){
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    auto bench = [&](){ reduce_by_cub_storage(d_temp_storage, temp_storage_bytes, begin, result.begin(), num_items, init);};

    bench();
    // Allocate temporary storage
    RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, 0));
    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);

    // Free temporary storage
    RMM_TRY(RMM_FREE(d_temp_storage, 0));
  }else{
    auto bench = [&](){ reduce_by_cub(result.begin(), begin, num_items, init);};

    bench(); // warm up

    do{
      BenchMarkTimer timer(iters);
      for (int i = 0; i < iters; ++i) {
        bench();
      }
    }while(0);
  }
};





// -----------------------------------------------------------------------------

struct benchmark
{
  template <typename T>
  void operator()(gdf_size_type column_size, int iters)
  {
    cudf::test::column_wrapper<T> hasnull_F(
      column_size,
      [](gdf_index_type row) { return T(row); });

    cudf::test::column_wrapper<T> hasnull_T(
      column_size,
      [](gdf_index_type row) { return T(row); },
      [](gdf_index_type row) { return row % 2 == 0; });

    rmm::device_vector<T> dev_result(1, T{0});

    bool no_new_allocate = false;

    do{
      std::cout << "new allocation: " << no_new_allocate << std::endl;

      raw_stream_bench_cub<T>(hasnull_F, dev_result, iters, no_new_allocate);
      iterator_bench_cub<T, false>(hasnull_F, dev_result, iters, no_new_allocate);
      iterator_bench_cub<T, true >(hasnull_T, dev_result, iters, no_new_allocate);

      no_new_allocate = !no_new_allocate;
    }while (no_new_allocate);

  };
};

void benchmark_types(gdf_size_type column_size, int iters, gdf_dtype type=N_GDF_TYPES)
{
  std::vector<gdf_dtype> types{};
  if (type == N_GDF_TYPES)
    types = {GDF_INT8, GDF_INT16, GDF_INT32, GDF_INT64, GDF_FLOAT32, GDF_FLOAT64};
  else
    types = {type};

  std::cout <<  "Iterator performance test:" << std::endl;
  std::cout <<  "  column_size = " << column_size << std::endl;
  std::cout <<  "  num iterates = " << iters << std::endl << std::endl;

  for (gdf_dtype t : types) {
    std::cout << name_from_type(t) << std::endl;
    cudf::type_dispatcher(t, benchmark(), column_size, iters);
    std::cout << std::endl << std::endl;
  }
}

int main(int argc, char **argv)
{
  gdf_size_type column_size{42000000};
  int iters{1000};
  gdf_dtype type = N_GDF_TYPES;

  if (argc > 1) column_size = std::stoi(argv[1]);
  if (argc > 2) iters = std::stoi(argv[2]);
  if (argc > 3) type = type_from_name(argv[3]);

  rmmOptions_t options{PoolAllocation, 0, false};
  rmmInitialize(&options);

  // -----------------------------------
  // type = GDF_FLOAT64;
  benchmark_types(column_size, iters, type);
  // -----------------------------------

  rmmFinalize();

  return 0;
}
