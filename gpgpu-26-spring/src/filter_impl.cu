#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include "curand_kernel.h"
#include "filter_impl.h"

#include <iostream>
#include <ostream>

#include "logo.h"

#define LOW 30
#define HIGH 40
#define RADIUS 1
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2 * RADIUS)
#define cudaCheckError() {                                                                   \
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

struct rgb
{
    uint8_t r, g, b;
};

struct __align__(8) reservoir
{
    rgb rgbV;
    unsigned int w;
};

const int K = 5;
const int MAX_WEIGHTS = 100;
const int THRESHOLD = 30;

reservoir* rs = nullptr;
uint8_t* dBuffer = nullptr;
bool* marker = nullptr;
bool* candidate = nullptr;
bool* d_changed = nullptr;
uint8_t* dOriginal = nullptr;
int* d_count = nullptr;
curandState* d_rand_states = nullptr;

static int res_width = 0;
static int res_height = 0;
static curandState* rng_states = nullptr;
__constant__ uint8_t* logo;

__global__ void masquage(uint8_t* input,uint8_t*mask, int width, int height,int input_stride,int mask_stride, int pixel_stride)
{
    rgb red = {255,0,0};
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    uint8_t* lineptr = input + y * input_stride + x * pixel_stride;
    uint8_t* maskptr = mask + y * mask_stride + x * pixel_stride;

    // partie rouge mis entre 0 et 1 (facteur)
    float m = maskptr[0] / 255.0f;

    lineptr[0] = (uint8_t)min(255.0f,(lineptr[0] + 0.5f * red.r * m));

}

__global__ void init_rng(curandState* states, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(idx, 0, 0, &states[idx]);
}


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

__device__ int matching_reservoir(rgb p, reservoir *res, int width, int height, int pitch_rs)
{
    int empty = -1;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size = pitch_rs;
    int idx = y * width + x;

    for (int j = 0; j < K; j++) {
        reservoir r = *(reservoir*)((uint8_t*)res+j*size+idx * sizeof(reservoir));
        if (r.w == 0)
        {
            if (empty == -1) {
                empty = j;
            }
            continue;
        }
        int dr = abs((int)p.r - (int)r.rgbV.r);
        int dg = abs((int)p.g - (int)r.rgbV.g);
        int db = abs((int)p.b - (int)r.rgbV.b);
        if (dr + dg + db < THRESHOLD)
        {
            return j;
        }
    }
    return empty;
}

__device__ uint8_t get_gray_pixel(uint8_t* buffer, int x, int y, int stride, int pixel_stride)
{
    uint8_t* pixel = buffer + y * stride + x * pixel_stride;
    return pixel[0];
}

__global__ void opening_kernel_shared(uint8_t* input, uint8_t* output, int width, int height, int stride, int pixel_stride)
{
    __shared__ uint8_t tile[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t erodeTile[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    int gx = min(max(x, 0), width - 1);
    int gy = min(max(y, 0), height - 1);

    tile[ty + RADIUS][tx + RADIUS] =
        input[gy * stride + gx * pixel_stride];

    if (tx < RADIUS)
    {
        int xx = max(x - RADIUS, 0);

        tile[ty + RADIUS][tx] =
            input[gy * stride + xx * pixel_stride];
    }

    if (tx >= BLOCK_SIZE - RADIUS)
    {
        int xx = min(x + RADIUS, width - 1);

        tile[ty + RADIUS][tx + 2 * RADIUS] =
            input[gy * stride + xx * pixel_stride];
    }

    if (ty < RADIUS)
    {
        int yy = max(y - RADIUS, 0);

        tile[ty][tx + RADIUS] =
            input[yy * stride + gx * pixel_stride];
    }

    if (ty >= BLOCK_SIZE - RADIUS)
    {
        int yy = min(y + RADIUS, height - 1);

        tile[ty + 2 * RADIUS][tx + RADIUS] =
            input[yy * stride + gx * pixel_stride];
    }

    if (tx < RADIUS && ty < RADIUS)
    {
        tile[ty][tx] =
            input[max(y - RADIUS, 0) * stride +
                  max(x - RADIUS, 0) * pixel_stride];
    }

    if (tx >= BLOCK_SIZE - RADIUS && ty < RADIUS)
    {
        tile[ty][tx + 2 * RADIUS] =
            input[max(y - RADIUS, 0) * stride +
                  min(x + RADIUS, width - 1) * pixel_stride];
    }

    if (tx < RADIUS && ty >= BLOCK_SIZE - RADIUS)
    {
        tile[ty + 2 * RADIUS][tx] =
            input[min(y + RADIUS, height - 1) * stride +
                  max(x - RADIUS, 0) * pixel_stride];
    }

    if (tx >= BLOCK_SIZE - RADIUS &&
        ty >= BLOCK_SIZE - RADIUS)
    {
        tile[ty + 2 * RADIUS][tx + 2 * RADIUS] =
            input[min(y + RADIUS, height - 1) * stride +
                  min(x + RADIUS, width - 1) * pixel_stride];
    }

    __syncthreads();

    if (x < width && y < height)
    {
        uint8_t v = 255;

        #pragma unroll
        for (int dy = -1; dy <= 1; dy++)
        {
            #pragma unroll
            for (int dx = -1; dx <= 1; dx++)
            {
                v = min(v,
                        tile[ty + RADIUS + dy]
                            [tx + RADIUS + dx]);
            }
        }

        erodeTile[ty][tx] = v;
    }

    __syncthreads();


    if (x < width && y < height)
    {
        uint8_t v = 0;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int xx = min(max(tx + dx, 0),
                             BLOCK_SIZE - 1);

                int yy = min(max(ty + dy, 0),
                             BLOCK_SIZE - 1);

                v = max(v, erodeTile[yy][xx]);
            }
        }

        int idx = y * stride + x * pixel_stride;

        output[idx] = v;
        output[idx + 1] = v;
        output[idx + 2] = v;
    }
}


/// @brief Initialise un état cuRAND par pixel (à lancer une seule fois)
/// @param states tableau de width * height états
/// @param width
/// @param height
/// @param seed graine globale
__global__ void init_rand_states(curandState* states, int width, int height,
                                 unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    curand_init(idx, 0, 0, &states[idx]);
}


__global__ void difference_kernel(uint8_t* buffer, reservoir* reservoirs, curandState* states,
                                  int width, int height, int stride,
                                  int pixel_stride, size_t pitch_rs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    uint8_t* line_ptr = buffer + y * stride + x * pixel_stride;
    rgb p = { line_ptr[0], line_ptr[1], line_ptr[2] };

    int m_idx = matching_reservoir(p, reservoirs, width, height, static_cast<int>(pitch_rs));


    float rand_val = curand_uniform(&states[idx]);

  
    int global_idx = m_idx*pitch_rs+idx*sizeof(reservoir);
    reservoir r = *(reservoir*)((uint8_t*)reservoirs+global_idx);
    if (m_idx != -1 && r.w > 0)
    {
        unsigned int w = r.w;
        if (w < MAX_WEIGHTS)
        {
            r.w++;
            w = r.w;
            r.rgbV.r = (uint8_t)(((unsigned int)r.rgbV.r * (w - 1) + p.r) / w);
            r.rgbV.g = (uint8_t)(((unsigned int)r.rgbV.g * (w - 1) + p.g) / w);
            r.rgbV.b = (uint8_t)(((unsigned int)r.rgbV.b * (w - 1) + p.b) / w);
        }
        else
        {
            r.rgbV.r = (uint8_t)(((unsigned int)r.rgbV.r * (MAX_WEIGHTS - 1) + p.r) / MAX_WEIGHTS);
            r.rgbV.g = (uint8_t)(((unsigned int)r.rgbV.g * (MAX_WEIGHTS - 1) + p.g) / MAX_WEIGHTS);
            r.rgbV.b = (uint8_t)(((unsigned int)r.rgbV.b * (MAX_WEIGHTS - 1) + p.b) / MAX_WEIGHTS);
        }
        *(reservoir*)((uint8_t*)reservoirs+global_idx) = r;
        line_ptr[0] = 0;
        line_ptr[1] = 0;
        line_ptr[2] = 0;
    }
    else if (m_idx != -1 && r.w == 0)
    {
        r.rgbV = p;
        r.w = 1;
        *(reservoir*)((uint8_t*)reservoirs+global_idx) = r;
    }
    else // Cas 3 : aucune correspondance, aucun slot vide
    {
        int min_idx = 0;

        for (int i = 1; i < K; i++) {
            reservoir* r1 = (reservoir*)((uint8_t*)reservoirs+pitch_rs*i+idx*sizeof(reservoir));
            reservoir* r2 = (reservoir*)((uint8_t*)reservoirs+pitch_rs*min_idx+idx*sizeof(reservoir));
            if (r1->w < r2->w)
                min_idx = i;
        }
        unsigned int total_w = 0;
        for (int i = 0; i < K; i++) {
            reservoir* r1 = (reservoir*)((uint8_t*)reservoirs+pitch_rs*i+idx*sizeof(reservoir));
            total_w += r1->w;
        }

        reservoir* r2 = (reservoir*)((uint8_t*)reservoirs+pitch_rs*min_idx+idx*sizeof(reservoir));
        if (rand_val * total_w >= r2->w)
        {
            r2->rgbV = p;
            r2->w = 1;
        }
    }

#define BG_MIN_WEIGHT 30
    int score = 255;
    bool found_established = false;

    for (int i = 0; i < K; i++)
    {
        reservoir r1 = *(reservoir*)((uint8_t*)reservoirs+pitch_rs*i+idx*sizeof(reservoir));
        if (r1.w >= BG_MIN_WEIGHT)
        {
            found_established = true;
            int d = max(
                abs((int)p.r - (int)r1.rgbV.r),
                max(
                    abs((int)p.g - (int)r1.rgbV.g),
                    abs((int)p.b - (int)r1.rgbV.b)
                )
            );
            score = min(score, d);
        }
    }

    if (!found_established)
    {
        int max_idx = 0;
        for (int i = 1; i < K; i++) {
            reservoir* r1 = (reservoir*)((uint8_t*)reservoirs+pitch_rs*i+idx*sizeof(reservoir));
            reservoir* r2 = (reservoir*)((uint8_t*)reservoirs+pitch_rs*max_idx+idx*sizeof(reservoir));
            if (r1->w > r2->w)
                max_idx = i;
        }

        reservoir* r2 = (reservoir*)((uint8_t*)reservoirs+pitch_rs*max_idx+idx*sizeof(reservoir));
        rgb bg = r2->rgbV;
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
__global__ void hysteresis_propagation(uint8_t* buffer, const bool* marker, const bool* candidate, int width, int height, size_t stride, int pixel_stride, int* changed_count) {
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

        atomicAdd(changed_count, 1);
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

                atomicAdd(changed_count, 1);
                break;
            }
        }
    }
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
                int pixel_stride, size_t pitch_rs)
{
    dim3 blockSize(32, 32);
    dim3 gridSize((width + 31)/32, (height + 31)/32);

    difference_kernel <<<gridSize, blockSize>>> (buffer, rs, d_rand_states, width, height, stride, pixel_stride, pitch_rs);
}

void cleanup() {
    if (rs != nullptr)
    {
        cudaFree(rs);
        rs = nullptr;
    }
    if (dBuffer != nullptr) {
        cudaFree(dBuffer);
        dBuffer = nullptr;
    }
    if (marker != nullptr) {
        cudaFree(marker);
        marker = nullptr;
    }
    if (candidate != nullptr) {
        cudaFree(candidate);
        candidate = nullptr;
    }
    if (d_changed != nullptr) {
        cudaFree(d_changed);
        d_changed = nullptr;
    }
    if (dOriginal != nullptr) {
        cudaFree(dOriginal);
        dOriginal = nullptr;
    }
    if (d_count != nullptr) {
        cudaFree(d_count);
        d_count = nullptr;
        if (d_rand_states != nullptr) {
            cudaFree(d_rand_states);
            d_rand_states = nullptr;
        }
    }
}

extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        static bool registered = false;
        assert(sizeof(rgb) == pixel_stride);
        static size_t pitch;
        static size_t original_pitch;
        static size_t pitch_rs;

        cudaError_t err;
        if (!registered)
        {
            atexit(cleanup);
            registered = true;
            err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
            CHECK_CUDA_ERROR(err);
            CHECK_CUDA_ERROR(cudaMalloc(&marker, width * height * sizeof(bool)));
            CHECK_CUDA_ERROR(cudaMalloc(&candidate, width * height * sizeof(bool)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_changed, sizeof(bool)));

            err = cudaMallocPitch(&dOriginal, &original_pitch, width * sizeof(rgb), height);
            CHECK_CUDA_ERROR(err);
            CHECK_CUDA_ERROR(cudaMalloc(&d_count, sizeof(int)));

            // Init cuRAND : un état par pixel, initialisé une seule fois
            CHECK_CUDA_ERROR(cudaMalloc(&d_rand_states, width * height * sizeof(curandState)));
            dim3 initBlock(32, 32);
            dim3 initGrid((width + 31) / 32, (height + 31) / 32);
            init_rand_states<<<initGrid, initBlock>>>(d_rand_states, width, height, 1234ULL);
            cudaCheckError();

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        dim3 blockSize(32,32);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

        // load_logo();
        if (rs == nullptr || res_width == 0 || res_height == 0)
        {
            if (rs != nullptr)
            {
                cudaFree(rs);
            }


            res_width = width;
            res_height = height;


            //cudaMallocPitch(&rs, pitch_rs, width * K * sizeof(reservoir), (unsigned int)height)
            CHECK_CUDA_ERROR(cudaMallocPitch(&rs, &pitch_rs, width * height * sizeof(reservoir), K));
            CHECK_CUDA_ERROR(
                cudaMemset2D(rs, pitch_rs, 0, width * height * sizeof(reservoir), K));

        }
        err = cudaMemcpy2D(dOriginal, original_pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dBuffer, pitch, dOriginal, original_pitch, width * sizeof(rgb), height, cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERROR(err);

        assert(sizeof(rgb) == pixel_stride);
        difference(dBuffer, width, height, pitch, pixel_stride, pitch_rs);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // STEP2: Ouverture
        dim3 blockSize2(16,16);
        dim3 gridSize2((width + (blockSize2.x - 1)) / blockSize2.x, (height + (blockSize2.y - 1)) / blockSize2.y);
        opening_kernel_shared<<<gridSize2, blockSize2>>>(dBuffer, dBuffer, width, height, pitch, pixel_stride);
        cudaDeviceSynchronize();
        cudaCheckError();

        // STEP 3 : Hysteresis
        hysteresis_init<<<gridSize, blockSize>>>(dBuffer, marker, candidate, width, height, pitch, pixel_stride);
        cudaDeviceSynchronize();
        cudaCheckError();

        int h_count = 1;
        while (h_count > 0)
        {
            CHECK_CUDA_ERROR(cudaMemset(d_count, 0, sizeof(int)));

            hysteresis_propagation<<<gridSize, blockSize>>>(dBuffer, marker, candidate, width, height, pitch, pixel_stride, d_count);
            cudaCheckError();

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        }
        //remove_red_channel_inp<<<gridSize, blockSize>>>(dBuffer, width, height, pitch);

        // STEP 4 : Masquage
        masquage<<<gridSize,blockSize>>>(dOriginal,dBuffer,width,height,original_pitch, pitch,pixel_stride);
        cudaDeviceSynchronize();
        cudaCheckError();

        err = cudaMemcpy2D(src_buffer, src_stride, dOriginal, original_pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);


    }

}