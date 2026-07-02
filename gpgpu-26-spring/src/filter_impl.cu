#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include "logo.h"

#define LOW 30
#define HIGH 40
#define cudaCheckError() {                                                                       \
    cudaError_t e=cudaGetLastError();                                                        \
    if(e!=cudaSuccess) {                                                                     \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
        exit(EXIT_FAILURE);                                                                  \
    }                                                                                        \
}

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

struct rgb {
    uint8_t r, g, b;
};

__constant__ uint8_t* logo;
/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer 
/// @param width 
/// @param height 
/// @param stride 
/// @param pixel_stride 
/// @return 
__global__ void remove_red_channel_inp(std::byte* buffer, int width, int height, int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    rgb* lineptr = (rgb*) (buffer + y * stride);
    if (y < logo_height && x < logo_width) {
        float alpha = logo[y * logo_width + x] / 255.f;
        lineptr[x].r = 0;
        lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1-alpha) * 255);
        lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1-alpha) * 255);
    } else {
        lineptr[x].r = 0;
    }
}

/// @brief Initialization for hysteresis filter on the image contained in "buffer"
/// @param buffer
/// @param marker should be of size width * height
/// @param candidate should be of size width * height
/// @param width
/// @param height
/// @param stride
/// @param pixel_stride
/// @return
__global__ void hysteresis_init(uint8_t* buffer, bool* marker, bool* candidate, int width, int height, size_t stride, int pixel_stride) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    uint8_t value = buffer[y * stride + x * pixel_stride];
    int i = y * width + x;
    candidate[i] = value >= LOW;
    marker[i] = value >= HIGH;

    buffer[y * stride + x * pixel_stride] = 0;
    buffer[y * stride + x * pixel_stride + 1] = 0;
    buffer[y * stride + x * pixel_stride + 2] = 0;
}

/// @brief Propagation for hysteresis filter on the image contained in "buffer"
/// @param buffer
/// @param marker should be of size width * height
/// @param candidate should be of size width * height
/// @param width
/// @param height
/// @param stride
/// @param pixel_stride
/// @param changed
/// @return
__global__ void hysteresis_propagation(uint8_t* buffer, const bool* marker, const bool* candidate, int width, int height, size_t stride, int pixel_stride, bool* changed) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    uint8_t value = buffer[y * stride + x * pixel_stride];
    int i = y * width + x;
    if (value) return;

    if (!candidate[i]) return;

    if (marker[i])
    {
        buffer[y * stride + x * pixel_stride] = 255;
        buffer[y * stride + x * pixel_stride + 1] = 255;
        buffer[y * stride + x * pixel_stride + 2] = 255;

        *changed = true;
        return;
    }

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;

            int yy = y + dy;
            int xx = x + dx;

            if (yy < 0 || yy >= height || xx < 0 || xx >= width) continue;

            if (buffer[yy * stride + xx * pixel_stride])
            {
                buffer[y * stride + x * pixel_stride] = 255;
                buffer[y * stride + x * pixel_stride + 1] = 255;
                buffer[y * stride + x * pixel_stride + 2] = 255;

                *changed = true;
                break;
            }
        }
    }
}


namespace
{
    void load_logo()
    {
        static auto buffer = std::unique_ptr<std::byte, decltype(&cudaFree)>{nullptr, &cudaFree}; 

        if (buffer == nullptr)
        {
            cudaError_t err;
            std::byte* ptr;
            err = cudaMalloc(&ptr, logo_width * logo_height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy(ptr, logo_data, logo_width * logo_height, cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpyToSymbol(logo, &ptr, sizeof(ptr));
            CHECK_CUDA_ERROR(err);

            buffer.reset(ptr);
        }

    }
}

extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        load_logo();

        assert(sizeof(rgb) == pixel_stride);
        uint8_t* dBuffer;
        size_t pitch;

        cudaError_t err;
        
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16,16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);



        // STEP 3 : Hysteresis
        bool* marker;
        err = cudaMalloc(&marker, width * sizeof(bool) * height);
        CHECK_CUDA_ERROR(err);

        bool* candidate;
        err = cudaMalloc(&candidate, width * sizeof(bool) * height);
        CHECK_CUDA_ERROR(err);

        hysteresis_init<<<gridSize, blockSize>>>(dBuffer, marker, candidate, width, height, pitch, pixel_stride);
        cudaDeviceSynchronize();
        cudaCheckError();

        bool* d_changed;
        err = cudaMalloc(&d_changed, sizeof(bool));
        CHECK_CUDA_ERROR(err);

        bool changed_host = true;
        while (changed_host) {
            changed_host = false;
            err = cudaMemset(&d_changed, changed_host, sizeof(bool));
            CHECK_CUDA_ERROR(err);

            hysteresis_propagation<<<gridSize, blockSize>>>(dBuffer, marker, candidate, width, height, pitch, pixel_stride, d_changed);
            cudaCheckError();

            err = cudaMemcpy(&changed_host, &d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
            CHECK_CUDA_ERROR(err);
        }
        //remove_red_channel_inp<<<gridSize, blockSize>>>(dBuffer, width, height, pitch);

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);
        cudaFree(marker);
        cudaFree(candidate);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);


        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
