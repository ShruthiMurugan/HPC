# Parallelization of K-Means Algorithm for MRI Dataset Analysis

## Introduction
The K-Means algorithm is commonly used for clustering analysis to identify groups of similar data points within a dataset. When dealing with large datasets, such as those in MRI (Magnetic Resonance Imaging) analysis, the serial implementation of K-Means can become computationally intensive. To address this issue and enhance computational efficiency, parallelization techniques utilizing frameworks such as CUDA can be employed.

This README provides an overview of the approach, aspects of parallelism, preprocessing steps, and the overall algorithm used for parallelizing the K-Means algorithm for MRI dataset analysis.

## Need for Parallelization
### Computationally Intensive: 
Dealing with large image datasets in MRI analysis requires significant computational resources.
### Enhanced Efficiency: 
Parallelization techniques, particularly using CUDA, can accelerate computations and optimize the overall process.
### Identifying Clusters: 
The K-Means algorithm is utilized to identify clusters of similar features within the MRI dataset, aiding in data analysis and interpretation.
## Aspects of Parallelism
### Data Parallelism: 
Distributing data across multiple processing units (e.g., GPU cores) to perform computations in parallel.
### Thread-Level Parallelism: 
Concurrent execution of threads within a single processing unit (e.g., GPU) to further accelerate computations.
## Preprocessing Steps
-Convert to Grayscale: Convert MRI images to grayscale to simplify analysis.

-Resizing: Resize images to a consistent dimension for uniform processing.

-Normalizing: Normalize pixel values to a standard range for consistency.

-Flatten and Store: Flatten images into a vector format and store them in a list or array.

-Convert to Numpy Array: Convert the list/array of images into a NumPy array for efficient processing.

-Write Pixel Values: Write pixel values of each image to a text file (e.g., 3ddata.txt) for input to the K-Means algorithm.

-Use Preprocessed Data: Utilize the preprocessed data as input for the K-Means clustering algorithm.
## Approach
### Input Preparation:
Take training input images and extract features, storing them in a list or array as a text file (e.g., 3ddata.txt).
### CUDA Initialization: 
Initialize the CUDA environment. Allocate memory for input data, centroids, clusters, and distances on the GPU (device memory) using CUDA memory management functions.
### Data Transfer: 
Copy the input data from the host (CPU) memory to the device (GPU) memory using cudaMemcpy.
### Parallel Execution: 
Define a CUDA kernel function (kmeans_kernel) responsible for calculating distances and assigning points to clusters in parallel. Configure grid and block dimensions for parallel execution.
### Kernel Computation:
Within the CUDA kernel, compute the Euclidean distance between data points and centroids, determine cluster assignments, and store distances and cluster assignments.
### Synchronization: 
Synchronize device threads to ensure completion of computations before proceeding.
### Performance Measurement: 
Print and measure execution time for different block sizes. Compute the speedup achieved by parallel execution.
### Memory Deallocation: 
Free allocated host and device memory using CUDA memory management functions.
