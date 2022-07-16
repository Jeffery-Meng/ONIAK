#include <falconn/lsh_nn_table.h>
#include "falconn/config.h"
#include "falconn/homogenizer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include "VectorUtils.hpp"

#include <AnnResultWriter.hpp>
#include <Exception.h>
#include <StringUtils.hpp>
#include <Timer.hpp>
#include <cstdio>

using namespace StringUtils;
const size_t MAX_MEM = 5e10; // 10 GB

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::make_pair;
using std::max;
using std::mt19937_64;
using std::pair;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using namespace VectorUtils;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::get_default_parameters;
using falconn::GaussianFunctionType;
using falconn::MultiProbeType;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborQuery;
using falconn::LSHNearestNeighborTable;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
using falconn::read_config;
using falconn::normalize_data;
using falconn::homogenize_data;
using falconn::queries_from_single_A;
using falconn::read_eigen_matrix;
using falconn::DenseMatrix;
using falconn::CoordinateType;

typedef DenseVector<float> Point;

const int NUM_QUERIES = 1000;
const int SEED = 4057218;
const int NUM_HASH_TABLES = 50;
const int NUM_HASH_BITS = 18;
const int NUM_ROTATIONS = 1;
const int BUCKET_ID_WIDTH = 10; // Not used

/*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */
bool read_point(FILE *file, Point *point) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  float *buf = new float[d];
  if (fread(buf, sizeof(float), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  point->resize(d + falconn::ADDITIONAL_DIMENSIONS);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(string file_name, vector<Point> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  Point p;
  dataset->clear();
  while (read_point(file, &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

/*
 * An auxiliary function that writes a dataset to a binary file 
 */
template <class T>
void write_datasets(string file_name, vector<vector<T>> dataset) {
  FILE *file = fopen(file_name.c_str(), "wb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  for (auto data:dataset)
  {
    int32_t d = data.size();
    fwrite(&d, sizeof(T), 1, file);
    fwrite(&data[0], sizeof(T), data.size(), file);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}


bool read_point_test(FILE *file, Point *point) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  float *buf = new float[d];
  if (fread(buf, sizeof(float), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  point->resize(d);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = buf[i];
  }
  delete[] buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset_test(string file_name, vector<Point> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  Point p;
  dataset->clear();
  while (read_point_test(file, &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  /* AIX and Solaris ------------------------------------------ */
  struct psinfo psinfo;
  int fd = -1;
  if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
    return (size_t)0L; /* Can't open? */
  if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
    close(fd);
    return (size_t)0L; /* Can't read? */
  }
  close(fd);
  return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))
  /* BSD, Linux, and OSX -------------------------------------- */
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
  return (size_t)rusage.ru_maxrss;
#else
  return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
  /* Unknown OS ----------------------------------------------- */
  return (size_t)0L; /* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &infoCount) != KERN_SUCCESS)
    return (size_t)0L; /* Can't access? */
  return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)
  /* Linux ---------------------------------------------------- */
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t)0L; /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t)0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
  /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
  return (size_t)0L; /* Unsupported. */
#endif
}

// Error message
#define NPP_ERROR_MSG(M)                                                       \
  do {                                                                         \
    fprintf(stderr, "%s:%d: " M, __FILE__, __LINE__);                          \
  } while (false)

// print parameters to stdout
void show_params(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  while (*fmt != '\0') {
    char *name = va_arg(args, char *);
    if (*fmt == 'i') {
      int val = va_arg(args, int);
      printf("%s: %d\n", name, val);
    } else if (*fmt == 'c') {
      int val = va_arg(args, int);
      printf("%s: \'%c\'\n", name, val);
    } else if (*fmt == 'f') {
      double val = va_arg(args, double);
      printf("%s: %f\n", name, val);
    } else if (*fmt == 's') {
      char *val = va_arg(args, char *);
      printf("%s: \"%s\"\n", name, val);
    } else {
      NPP_ERROR_MSG("Unsupported format");
    }
    ++fmt;
  }

  va_end(args);
}

void usage() {
  printf("Falconn\n");
  printf("Options\n");
  printf("-d {value}     \trequired \tdimensionality\n");
  printf("-ds {string}   \trequired \tdataset file\n");
  printf("-n {value}     \trequired \tcardinality\n");
  printf("-l {value}     \trequired \tparameter l (# of hash tables)\n");
  printf("-m {value}     \trequired \tparameter M (# of hash "
         "functions)\n");
  printf("-w {value}     \trequired \tparameter W (bucket_width) \n");
  printf("-u {value}     \trequired \tparameter U (dataset universe)\n");
  printf("-t {value}     \trequired \tparameter t (the number "
         "of probes)\n");
  printf("-k {value}     \toptional \tnumber of neighbors "
         "(default: 1)\n");
  printf("-gt {string}   \trequired  \tfile of exact results\n");
  printf("-qs {string}   \trequired \tfile of query set\n");
  printf("-qn {string}   \trequired \tnumber of queries\n");
  printf("-rf {string}   \trequired \tresult file\n");
  printf("-if {string}   \trequired \tindex folder\n");

  printf("\n");
  printf("Run falconn (indexing and querying)\n");
  printf("-d -n -ds -l -t -m -u -w -gt -qs -qn -rf -if [-k]\n");

  printf("\n");
}

std::unique_ptr<LSHNearestNeighborTable<Point>>
indexing(const LSHConstructionParameters &params,  // config filename
         std::vector<Point> &train
        ) { 
  read_dataset(params.data_filename, &train);
  std::cout << "Maximum partition metric is: " << falconn::maximum_partition_metric(train)<< std::endl;
//homogenize_data(train, params.dimension-1);
  falconn::global::filtered_keys = normalize_data(train, params);
  
  

  // setting parameters and constructing the table
  
  /*params.dimension = dim;
  params.lsh_family = LSHFamily::Gaussian;
  params.k = num_hashfuncs;
  params.l = num_hashes;
  params.distance_function = DistanceFunction::L1Norm;
  // compute_number_of_hash_functions<Point>(num_hashbits, &params);
  params.multi_probe = MultiProbeType::Precomputed;
  params.gauss_type = GaussianFunctionType::Cauchy;
  params.num_setup_threads = 1;
  params.storage_hash_table = StorageHashTable::FlatHashTable;
  params.bucket_id_width = BUCKET_ID_WIDTH;
  params.bucket_width = bucket_width*1.0;
  params.universe = universe;*/

  std::string perf_filename = params.index_record_path + join(
      {"n" + std::to_string(params.num_points), "d" + std::to_string(params.dimension),
       "l" + std::to_string(params.hash_table_params[0].l), "m" + std::to_string(params.hash_table_params[0].k)},
      "-");
  perf_filename += ".txt";

  AnnResultWriter writer(params.index_path + perf_filename, params.allow_overwrite);
  writer.writeRow(
      "s",
      "dsname,#n,#dim,#hashes,#functions,index_size(bytes),construction_time(us)");
  const char *fmt = "siiiiif";

  HighResolutionTimer timer;
  timer.restart();
  auto table = construct_table<Point>(train, params);
  auto e = timer.elapsed();

  auto isz = getPeakRSS();

  writer.writeRow(fmt, params.data_filename.c_str(), params.num_points, params.dimension, params.hash_table_params[0].l, params.hash_table_params[0].k, isz, e);

  return table;
}

void knn(LSHNearestNeighborTable<Point> *table, const std::vector<Point> &train,
         const LSHConstructionParameters &params) {
  NPP_ENFORCE(table != nullptr);
  std::vector<Point> test;
  read_dataset_test(params.query_filename, &test);
  NPP_ENFORCE(test.size() == params.num_queries);
  NPP_ENFORCE(test.size() == params.num_queries);
  

  std::ifstream kernel_in(params.kernel_filename, std::ios::binary);
  std::ifstream query_in(params.query_filename, std::ios::binary);
  //auto kernel = read_eigen_matrix<falconn::CoordinateType>(fin, params.dim_Arows, 
  //                                            params.dimension-falconn::ADDITIONAL_DIMENSIONS);
  auto queries = falconn::read_one_matrix<float>(query_in, params.num_queries, params.raw_dimension);
  int qn = params.num_queries, K = params.num_neighbors;
  std::ifstream gt_in(params.gnd_filename);

  /*unsigned r_qn, r_maxk;
  FILE *fp = fopen(params.gnd_filename.c_str(), "r");
  NPP_ENFORCE(fp != NULL);
  NPP_ENFORCE(fscanf(fp, "%d %d\n", &r_qn, &r_maxk) >= 0);
  
  NPP_ENFORCE(r_qn >= params.num_queries && r_maxk >= K);

  std::vector<int> gt(qn * r_maxk, -1.0f);

  for (unsigned i = 0; i < qn; ++i) {
    unsigned j;
    NPP_ENFORCE(fscanf(fp, "%d", &j) >= 0);
    NPP_ENFORCE(j == i);
    for (j = 0; j < r_maxk; ++j) {
      NPP_ENFORCE(fscanf(fp, " %d", &gt[i * r_maxk + j]) >= 0);
    }
    NPP_ENFORCE(fscanf(fp, "\n") >= 0);
  }
  NPP_ENFORCE(fclose(fp) == 0);

  auto ground_idx = to_matrix(gt, qn, r_maxk);*/
  auto ground_idx = falconn::read_ground_truth(gt_in, K, qn);

  
  HighResolutionTimer timer;
  AnnResultWriter writer(params.result_filename, params.allow_overwrite);

  std::string summary_path = params.summary_path + join(
      {"w" + std::to_string(params.hash_table_params[0].bucket_width), 
      "m" + std::to_string(params.hash_table_params[0].k), 
      "l" + std::to_string(params.hash_table_params[0].l)},
      "-");
  
  CppFileHelper summary_file(summary_path);
  double candidate_sum = 0., recall_sum = 0., query_time_sum = 0., query_raw_sum = 0.;


  writer.writeRow("s", AnnResults::_DEFAULT_HEADER_I_);
  //unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
  auto query_object_ori =    table->construct_query_object(
              params.hash_table_params[0].l * falconn::PROBES_PER_TABLE, -1, 1, 3);
  auto *query_object = query_object_ori.get();

  // For-loop over matrix A, when A changed, A_changed set to True.

  DenseMatrix<CoordinateType> kernel, Ay_query;
  kernel = falconn::read_one_matrix<float>(kernel_in, params.dim_Arows, params.raw_dimension);
  std::vector<std::vector<int32_t>> total_res;
  for (unsigned i = 0; i < qn; i++) {
    if (i > 0 && !params.single_kernel) {  // read a new kernel from stream
      kernel = falconn::read_one_matrix<float>(kernel_in, params.dim_Arows, params.raw_dimension);
    }

    if (params.transformed_queries) {
      Ay_query = falconn::combine_one(kernel, queries.row(i).transpose());
    } else {
      Ay_query = falconn::combine_one_transform(kernel, queries.row(i).transpose());
    }
    
    std::vector<int32_t> res;
    
#ifdef DEBUG
    printf("Query startes for %d\n", i);
#endif
    timer.restart();
    query_object->reset_query_statistics();
#ifdef DEBUG
    printf("Perform query for %d\n", i);
#endif
 
    query_object->get_unique_candidates(Ay_query, &res);
   //std::cout << "Num of neighbors: " << K << "result size: " << res.size() << std::endl;
    auto query_time = timer.elapsed();
    // std::ofstream fout("result_candidates.txt");
    // for(auto item:res) {
    //  fout<<res << std::endl;
    // }
    // fout.close();
    total_res.push_back(res);

    const std::vector<int>& ground_truth = ground_idx[i];
    std::sort(res.begin(), res.end());
    std::vector<int> v_intersection;
    std::set_intersection(ground_truth.begin(),
       ground_truth.end(), res.begin(), res.end(), std::back_inserter(v_intersection));
    double recall = (float) v_intersection.size() / (float) K;

  auto statistics = query_object->get_query_statistics();
    double candi_num = statistics.average_num_unique_candidates;
    //std::cout << candi_num << std::endl;
    writer.writeRow("ifff", i,  candi_num, recall,
                        query_time);
    candidate_sum += candi_num;
    recall_sum += recall;
    query_raw_sum += query_time;
    query_time_sum += statistics.average_total_query_time;


  }

  // Write candidates
  // std::string filename = params.summary_path+"candidates_index.ivecs";
  // write_datasets(filename,total_res);

  summary_file.print_one_line("l", "m", "w", "recall", "candidate ratio", "query time");
  summary_file.print_one_line(params.hash_table_params[0].l, params.hash_table_params[0].k, 
  params.hash_table_params[0].bucket_width, recall_sum / qn, 
  candidate_sum/((double)params.num_points*(double)qn), 
        query_time_sum/qn);
}

/*
 * Get the index of next unblank char from a string.
 */
int GetNextChar(char *str) {
  int rtn = 0;

  // Jump over all blanks
  while (str[rtn] == ' ') {
    rtn++;
  }

  return rtn;
}

/*
 * Get next word from a string.
 */
void GetNextWord(char *str, char *word) {
  // Jump over all blanks
  while (*str == ' ') {
    str++;
  }

  while (*str != ' ' && *str != '\0') {
    *word = *str;
    str++;
    word++;
  }

  *word = '\0';
}

int main(int argc, char **argv) {
  int cnt = 1;
  bool failed = false;
  char *arg;
  int i;
  char para[10];

  char conf[200] = "";

  std::string err_msg;
  while (cnt < argc && !failed) {
    arg = argv[cnt++];
    if (cnt == argc) {
      failed = true;
      break;
    }

    i = GetNextChar(arg);
    if (arg[i] != '-') {
      failed = true;
      err_msg = "Wrong format!";
      break;
    }

    GetNextWord(arg + i + 1, para);

    arg = argv[cnt++];

    if (strcmp(para, "cf") == 0) {
      GetNextWord(arg, conf);
    } else {
      failed = true;
      fprintf(stderr, "Unknown option -%s!\n\n", para);
    }
  }

  if (failed) {
    fprintf(stderr, "%s:%d: %s\n\n", __FILE__, __LINE__, err_msg.c_str());
    usage();
    return EXIT_FAILURE;
  }


  std::vector<Point> train;


    LSHConstructionParameters params = read_config(conf);
    #ifndef DISABLE_VERBOSE
  printf("=====================================================\n");
  show_params("iiiiiiifssss", "# of points", params.num_points, "raw dimension",
             params.raw_dimension, "# of queries", params.num_queries, "# of hash tables", params.hash_table_params[0].l,
             "# of hash functions", params.hash_table_params[0].k, "# of probes per table", falconn::PROBES_PER_TABLE, "k", params.num_neighbors,
             "bucket width",params.hash_table_params[0].bucket_width, 
             "dataset filename", params.data_filename.c_str(), "configuration path",conf, "result filename",
             params.result_filename.c_str(), "ground truth filename", params.gnd_filename.c_str());
  printf("=====================================================\n");
#endif


    auto table = indexing(params, train);

    knn(&*table, train, params);

 // } catch (const npp::Exception &e) {
  //  std::cerr << e.what() << std::endl;
 ///// } catch (const std::exception &e) {
 //   std::cerr << e.what() << std::endl;
 // }

  return EXIT_SUCCESS;
}
