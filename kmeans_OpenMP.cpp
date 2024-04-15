#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <random>

void kmeans_kernel(float *data, float *centroids, int *clusters, float *distances, int N, int K, int D) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float min_dist = INFINITY;
        int min_cluster = -1;
        for (int j = 0; j < K; j++) {
            float dist = 0;
            for (int d = 0; d < D; d++) {
                float diff = data[i * D + d] - centroids[j * D + d];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if (dist < min_dist) {
                min_dist = dist;
                min_cluster = j;
            }
        }
        clusters[i] = min_cluster;
        distances[i] = min_dist;
    }
}

int main() {
    int K = 2;
    int D = 3;
    int MAX_ITER = 100;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 100.0);

    std::vector<double> speedup;
    for (int N : {100000, 200000, 500000, 1000000, 2000000}) {
        std::vector<float> data_vec(N * D);
        for (int i = 0; i < N * D; ++i) {
            data_vec[i] = dis(gen);
        }

        float *data = data_vec.data();
        float *centroids = new float[K * D];
        int *clusters = new int[N];
        float *distances = new float[N];

        for (int i = 0; i < K * D; ++i) {
            centroids[i] = dis(gen);
        }

        double start_time, end_time;

        start_time = omp_get_wtime();
        for (int iter = 0; iter < MAX_ITER; iter++) {
            kmeans_kernel(data, centroids, clusters, distances, N, K, D);
        }
        end_time = omp_get_wtime();
        double milliseconds_single = (end_time - start_time) * 1000;

        std::cout << "Execution time for problem size " << N << " (single thread): " << milliseconds_single << " ms" << std::endl;

        for (int num_threads : {2, 4, 8, 16, 32}) {
            omp_set_num_threads(num_threads);

            start_time = omp_get_wtime();
            for (int iter = 0; iter < MAX_ITER; iter++) {
                kmeans_kernel(data, centroids, clusters, distances, N, K, D);
            }
            end_time = omp_get_wtime();
            double milliseconds = (end_time - start_time) * 1000;

            speedup.push_back(milliseconds_single / milliseconds);

            std::cout << "Execution time for problem size " << N << " with " << num_threads << " threads: " << milliseconds << " ms" << std::endl;
        }

        delete[] centroids;
        delete[] clusters;
        delete[] distances;
    }

    std::cout << "\nSpeedup for different problem sizes and number of threads:" << std::endl;
    for (size_t i = 0; i < speedup.size(); ++i) {
        std::cout << "Problem size " << (i / 5) << ", Threads " << (1 << (i % 5)) << ": " << speedup[i] << std::endl;
    }

    return 0;
}
