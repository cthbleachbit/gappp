## how to compile
nvcc -vectorAdd.cu -o add.o

## initialize cuda memory
```
// Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
```

## copy host memory to gpu
```
err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
```

## initialize cuda threads or blocks (that contain threads)
```
int threadsPerBlock = 256;
int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```
the thing is to add <<<blocksPerGrid, threadsPerBlock>>> before the cuda function.

## indexing inside of the cuda function

```
int i = blockDim.x * blockIdx.x + threadIdx.x;
```
if you have only the threads, then

```
int i = threadIdx.x;
```

## copy cuda memory to host memory
```
err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
```

## free cuda memory

```
err = cudaFree(d_C);
```
