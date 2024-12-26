#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <time.h>
#include <vector>

#define MAX_FRONTIER_SIZE 128

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }


#define THREADS_PER_BLOCK 32
#define BLOCKS_PER_GRID 256
#define MAX_N_NODES 383


// PROTOTYPES
void read_matrix(std::vector<int> &row_ptr, std::vector<int> &col_ind, std::vector<float> &values, const std::string &filename, int &num_rows, int &num_cols, int &num_vals);


// GPU KERNEL
__global__ void breadth_first_search(const int source, const int* row_list, const int* col_list, int* distance_list, int* current_frontier, int* next_frontier, int* update_frontier_vector, int n_nodes) {

  int current_size = 1;
  current_frontier[0] = source;

  int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ int local_frontier_matrix[THREADS_PER_BLOCK][MAX_N_NODES]; // [numero thread][numero nodi per thread (al max numero nodi)]
  __shared__ int local_frontier_size[THREADS_PER_BLOCK]; // [numero thread], cambia anche update_frontier_vector riga 65 e 89

  while(current_size > 0) {
    
    update_frontier_vector[thread_index] = 0;
    local_frontier_size[threadIdx.x] = 0;
    if(thread_index < current_size) {
      // actually compute the next frontier for node current_list[thread_index] in shared memory
      int node = current_frontier[thread_index];
      for (int neighbour = row_list[node]; neighbour < row_list[node + 1]; ++neighbour) {
        atomicCAS(&distance_list[col_list[neighbour]], distance_list[col_list[neighbour]] == -1, node + n_nodes);

        if(distance_list[col_list[neighbour]] == node + n_nodes) {
            distance_list[col_list[neighbour]] = distance_list[node] + 1;
            local_frontier_matrix[threadIdx.x][local_frontier_size[threadIdx.x]] = neighbour;
            local_frontier_size[threadIdx.x] += 1;
        }

      }

      // update update_frontier_vector: for each index i add the size of the local frontier of the thread,
      // so that eventually every value in update_frontier_vector will be the index in next_frontier to start to write from the local frontier
      for(int i = thread_index; i < current_size; ++i) {
        atomicAdd(&update_frontier_vector[i], local_frontier_size[threadIdx.x]);
      }

      __syncthreads();

      if(thread_index == 0) {
        for(int counter = 1; counter <= local_frontier_size[threadIdx.x]; counter++) {
          next_frontier[counter - 1] = local_frontier_matrix[threadIdx.x][counter - 1];
        }
      }
      else {
        for(int counter = 1; counter <= local_frontier_size[threadIdx.x]; counter++) {
          next_frontier[update_frontier_vector[thread_index - 1] + counter - 1] = local_frontier_matrix[threadIdx.x][counter - 1];
        }
      }

      __syncthreads();

      // only one thread has to swap the pointers and update the sizes
      if(thread_index == 0) {
        int* box = current_frontier;
        current_frontier = next_frontier;
        next_frontier = box;

        current_size = update_frontier_vector[current_size - 1]; // change with number of threads
      }

      __syncthreads();
    } 

  }
}

// MAIN
int main(int argc, char *argv[]) {

  if (argc != 3) {
    printf("Usage: ./exec matrix_file source\n");
    return 0;
  }
  const std::string filename{argv[1]};
  const int source = atoi(argv[2]) - 1; // The node starts from 1 but array starts from 0

  
  std::vector<int> row_ptr;
  std::vector<int> col_ind;
  std::vector<float> values;
  int num_rows, num_cols, num_vals;

  read_matrix(row_ptr, col_ind, values, filename, num_rows, num_cols, num_vals);

  // Initialize dist to -1
  std::vector<int> dist(num_vals);
  for (int i = 0; i < num_vals; i++) { dist[i] = -1; }


  // 1 ------ MEMCPY TO GPU
  /*
    Need to copy:
        1. the graph matrix : input
        2. the distance vector : output
  */
  int* row_list, *col_list, *distance_list_gpu;
  int* current_frontier, *next_frontier, *update_frontier_vector; // update_frontier_vector helps in adding next_frontier nodes to current_frontier vector

  CHECK(cudaMalloc(&row_list, row_ptr.size()*sizeof(int)));
  CHECK(cudaMemcpy(row_list, row_ptr.data(), row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&col_list, col_ind.size()*sizeof(int)));
  CHECK(cudaMemcpy(col_list, col_ind.data(), col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&distance_list_gpu, dist.size()*sizeof(int)));
  CHECK(cudaMemcpy(distance_list_gpu, dist.data(), dist.size()*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&current_frontier, num_vals*sizeof(int)));
  CHECK(cudaMalloc(&next_frontier, num_vals*sizeof(int)));
  CHECK(cudaMalloc(&update_frontier_vector, num_vals*sizeof(int))); // technically one for each thread, but for now it's fine with num_vals


  // 2 ------ KERNEL
  /*

    Need to create:
      1. vector for current frontier
      2. vector for next frontier
      3. shared memory vector for next local frontier

    Iterate until current frontier is empty
    
    Every thread should:
        1. pick up a new node in currentFrontier
        2. search for all neighbour nodes via CSR format, saving them on a local array in shared memory
        3. update distances array
        4. write on a global memory array the number of nodes found (global array of positions)
    
    When currentFrontier size is 0:
        1. sinchronize all threads
        2. copy each local array in next frontier, using the global array of positions to know where to start to copy
        3. switch the current frontier pointer with next frontier pointer (this can be done by just one thread)
  */

  // number of threads equal to number of nodes (?)
  breadth_first_search<<<THREADS_PER_BLOCK, BLOCKS_PER_GRID>>>(source, row_list, col_list, distance_list_gpu, current_frontier, next_frontier, update_frontier_vector, num_vals);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  // 3 ------ MEMCPY TO CPU
  std::vector<int> distance_list_cpu(dist.size());
  CHECK(cudaMemcpy(distance_list_cpu.data(), distance_list_gpu, dist.size()*sizeof(int), cudaMemcpyDeviceToHost));

  for(int value : distance_list_cpu) {
    printf("%d", value);
  }

  return EXIT_SUCCESS;
}



// FUNCTIONS


// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(std::vector<int> &row_ptr,
                 std::vector<int> &col_ind,
                 std::vector<float> &values,
                 const std::string &filename,
                 int &num_rows,
                 int &num_cols,
                 int &num_vals) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "File cannot be opened!\n";
    throw std::runtime_error("File cannot be opened");
  }

  // Get number of rows, columns, and non-zero values
  file >> num_rows >> num_cols >> num_vals;

  row_ptr.resize(num_rows + 1);
  col_ind.resize(num_vals);
  values.resize(num_vals);

  // Collect occurrences of each row for determining the indices of row_ptr
  std::vector<int> row_occurrences(num_rows, 0);

  int row, column;
  float value;
  while (file >> row >> column >> value) {
    // Subtract 1 from row and column indices to match C format
    row--;
    column--;

    row_occurrences[row]++;
  }

  // Set row_ptr
  int index = 0;
  for (int i = 0; i < num_rows; i++) {
    row_ptr[i] = index;
    index += row_occurrences[i];
  }
  row_ptr[num_rows] = num_vals;

  // Reset the file stream to read again from the beginning
  file.clear();
  file.seekg(0, std::ios::beg);

  // Read the first line again to skip it
  file >> num_rows >> num_cols >> num_vals;

  std::fill(col_ind.begin(), col_ind.end(), -1);

  int i = 0;
  while (file >> row >> column >> value) {
    row--;
    column--;

    // Find the correct index (i + row_ptr[row]) using both row information and an index i
    while (col_ind[i + row_ptr[row]] != -1) { i++; }
    col_ind[i + row_ptr[row]] = column;
    values[i + row_ptr[row]]  = value;
    i                         = 0;
  }
}
