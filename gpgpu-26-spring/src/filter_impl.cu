#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#include "filter_impl.h"
#include "logo.h"

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

struct rgb
{
    uint8_t r, g, b;
};

struct reservoir
{
    rgb rgbV;
    unsigned int w;
};

const int K = 5;
const int MAX_WEIGHTS = 100;
const int THRESHOLD = 30;

reservoir* rs = nullptr;
static int res_width = 0;
static int res_height = 0;

__constant__ uint8_t* logo;

/// @brief Black out the red channel from the video and add EPITA's logo
/// @param buffer
/// @param width
/// @param height
/// @param stride
/// @param pixel_stride
/// @return
__global__ void remove_red_channel_inp(std::byte* buffer, int width, int height,
                                       int stride)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return;

    rgb* lineptr = (rgb*)(buffer + y * stride);
    if (y < logo_height && x < logo_width)
    {
        float alpha = logo[y * logo_width + x] / 255.f;
        lineptr[x].r = 0;
        lineptr[x].g = uint8_t(alpha * lineptr[x].g + (1 - alpha) * 255);
        lineptr[x].b = uint8_t(alpha * lineptr[x].b + (1 - alpha) * 255);
    }
    else
    {
        lineptr[x].r = 0;
    }
}

__device__ int matching_reservoir(rgb p, reservoir* res)
{
    int empty = -1;
    for (int j = 0; j < K; j++)
    {
        if (res[j].w == 0)
        {
            if (empty == -1) {
                empty = j;
            }
            continue;
        }
        int dr = abs((int)p.r - (int)res[j].rgbV.r);
        int dg = abs((int)p.g - (int)res[j].rgbV.g);
        int db = abs((int)p.b - (int)res[j].rgbV.b);
        if (dr + dg + db < THRESHOLD)
        {
            return j;
        }
    }
    return empty;
}

__global__ void difference_kernel(uint8_t* buffer, reservoir* reservoirs,
                                  int width, int height, int stride,
                                  int pixel_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    uint8_t* line_ptr = buffer + y * stride + x * pixel_stride;
    rgb p = { line_ptr[0], line_ptr[1], line_ptr[2] };

    reservoir* res = reservoirs + idx * K;

    int m_idx = matching_reservoir(p, res);

    unsigned int seed = idx * 1234567 + threadIdx.x; // pour le rapport parler de ça ptetre
    float rand_val = (seed % 1000) / 1000.0f;

    if (m_idx != -1 && res[m_idx].w > 0)
    {
        unsigned int w = res[m_idx].w;
        if (w < MAX_WEIGHTS)
        {
            res[m_idx].w++;
            w = res[m_idx].w;
            res[m_idx].rgbV.r = (uint8_t)(((unsigned int)res[m_idx].rgbV.r * (w - 1) + p.r) / w);
            res[m_idx].rgbV.g = (uint8_t)(((unsigned int)res[m_idx].rgbV.g * (w - 1) + p.g) / w);
            res[m_idx].rgbV.b = (uint8_t)(((unsigned int)res[m_idx].rgbV.b * (w - 1) + p.b) / w);
        }
        else
        {
            res[m_idx].rgbV.r = (uint8_t)(((unsigned int)res[m_idx].rgbV.r * (MAX_WEIGHTS - 1) + p.r) / MAX_WEIGHTS);
            res[m_idx].rgbV.g = (uint8_t)(((unsigned int)res[m_idx].rgbV.g * (MAX_WEIGHTS - 1) + p.g) / MAX_WEIGHTS);
            res[m_idx].rgbV.b = (uint8_t)(((unsigned int)res[m_idx].rgbV.b * (MAX_WEIGHTS - 1) + p.b) / MAX_WEIGHTS);
        }

        line_ptr[0] = 0;
        line_ptr[1] = 0;
        line_ptr[2] = 0;
    }
    else if (m_idx != -1 && res[m_idx].w == 0)
    {
        res[m_idx].rgbV = p;
        res[m_idx].w = 1;
    }
    else // Cas 3 : aucune correspondance, aucun slot vide
    {
        int min_idx = min_reservoir(res);

        unsigned int total_w = 0;
        for (int i = 0; i < K; i++)
            total_w += res[i].w;

        if (rand_val * total_w >= res[min_idx].w)
        {
            res[min_idx].rgbV = p;
            res[min_idx].w = 1;
        }
    }

#define BG_MIN_WEIGHT 30
    int score = 255;
    bool found_established = false;

    for (int i = 0; i < K; i++)
    {
        if (res[i].w >= BG_MIN_WEIGHT)
        {
            found_established = true;
            int d = max(
                abs((int)p.r - (int)res[i].rgbV.r),
                max(
                    abs((int)p.g - (int)res[i].rgbV.g),
                    abs((int)p.b - (int)res[i].rgbV.b)
                )
            );
            score = min(score, d);
        }
    }

    if (!found_established)
    {
        int max_idx = 0;
        for (int i = 1; i < K; i++)
            if (res[i].w > res[max_idx].w)
                max_idx = i;

        rgb bg = res[max_idx].rgbV;
        score = max(
            abs((int)p.r - (int)bg.r),
            max(
                abs((int)p.g - (int)bg.g),
                abs((int)p.b - (int)bg.b)
            )
        );
    }

    uint8_t value = (uint8_t)min(score, 255);
    line_ptr[0] = value;
    line_ptr[1] = value;
    line_ptr[2] = value;
}

namespace
{
    void load_logo()
    {
        static auto buffer =
            std::unique_ptr<std::byte, decltype(&cudaFree)>{ nullptr,
                                                             &cudaFree };

        if (buffer == nullptr)
        {
            cudaError_t err;
            std::byte* ptr;
            err = cudaMalloc(&ptr, logo_width * logo_height);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpy(ptr, logo_data, logo_width * logo_height,
                             cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);

            err = cudaMemcpyToSymbol(logo, &ptr, sizeof(ptr));
            CHECK_CUDA_ERROR(err);

            buffer.reset(ptr);
        }
    }
} // namespace

void difference(uint8_t* buffer, int width, int height, int stride,
                int pixel_stride)
{
    uint8_t* dev_buffer;
    cudaMalloc(&dev_buffer, height * stride);
    cudaMemcpy(dev_buffer, buffer, height * stride, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15)/16, (height + 15)/16);

    difference_kernel <<<gridSize, blockSize>>> (dev_buffer, rs, width, height, stride, pixel_stride);

    cudaMemcpy(buffer, dev_buffer, height * stride, cudaMemcpyDeviceToHost);
    cudaFree(dev_buffer);
}


    extern "C"
{
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride,
                     int pixel_stride)
    {
        load_logo();

        if (rs == nullptr || res_width == 0 || res_height == 0)
        {
            if (rs != nullptr)
            {
                cudaFree(rs);
            }

            res_width = width;
            res_height = height;

            CHECK_CUDA_ERROR(
                cudaMalloc(&rs, width * height * K * sizeof(reservoir)));
            CHECK_CUDA_ERROR(
                cudaMemset(rs, 0, width * height * K * sizeof(reservoir)));
        }

        assert(sizeof(rgb) == pixel_stride);
        difference(src_buffer, width, height, src_stride, pixel_stride);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        {
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(100ms);
        }
    }
}
