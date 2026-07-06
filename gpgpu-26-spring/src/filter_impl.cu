#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>

#include "filter_impl.h"
#include "logo.h"
#include "curand_kernel.h"

#define LOW 30
#define HIGH 40
#define RADIUS 1
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

static curandState* rng_states = nullptr;

__global__ void init_rng(curandState* states, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(idx, 0, 0, &states[idx]);
}

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

__device__ uint8_t get_gray_pixel(uint8_t* buffer, int x, int y, int stride, int pixel_stride)
{
    uint8_t* pixel = buffer + y * stride + x * pixel_stride;
    return pixel[0];
}
__global__ void erosion_kernel(uint8_t* buffer,uint8_t * eroded,  int width, int height, int stride,int pixel_stride,int radius)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }
    uint8_t min_value = 255;
    for (int dy = -radius; dy <= radius; dy++)
    {
	    int yy = y + dy;
	    if (yy < 0 || yy >= height)
	    {
	    	continue;
	    }
	    for (int dx = -radius; dx <= radius;dx++)
	    {
		    int xx = x + dx;
		    if (xx < 0 || xx >= width)
		    {
			    continue;
		    }
		    
		    uint8_t value = get_gray_pixel(buffer,xx,yy,stride,pixel_stride); 
		    min_value = min(min_value,value);
	    }
    }
    eroded[y * width + x] = min_value;    
}


__global__ void dilatation_kernel(const uint8_t * input,uint8_t* output, int width, int height, int stride, int pixel_stride,int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    uint8_t max_value = 0;
    for (int dy = -radius; dy <= radius; dy++)
    {
	    int yy = y + dy;
	    if (yy < 0 || yy >= height)
	    {
	    	continue;
	    }
	    for (int dx = -radius; dx <= radius;dx++)
	    {
		    int xx = x + dx;
		    if (xx < 0 || xx >= width)
		    {
			    continue;
		    }
		    
		    uint8_t value = input[yy * width + xx]; 
		    max_value = max(max_value,value);
	    }
    }
    int idx =  y * stride + x * pixel_stride;
    output[idx] = max_value;
    output[idx + 1] = max_value;
    output[idx + 2] = max_value;
     
}


__global__ void difference_kernel(uint8_t* buffer, reservoir* reservoirs,
                                  curandState* states,
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

    float rand_val = curand_uniform(&states[idx]);
    
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
        int min_idx = 0;
        for (int i = 1; i < K; i++)
            if (res[i].w < res[min_idx].w)
                min_idx = i;
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

    difference_kernel<<<gridSize, blockSize>>>(dev_buffer, rs, rng_states,
                                               width, height, stride, pixel_stride);

    cudaMemcpy(buffer, dev_buffer, height * stride, cudaMemcpyDeviceToHost);
    cudaFree(dev_buffer);
}

void cleanup()
{
    if (rs != nullptr)
    {
        cudaFree(rs);
        rs = nullptr;
        cudaFree(rng_states);
        rng_states = nullptr;
    }
}


extern "C" 
{
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        static bool registered = false;
        if (!registered)
        {
            atexit(cleanup);
            registered = true;
        }

        load_logo();

        dim3 blockSize(16,16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

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

            CHECK_CUDA_ERROR(cudaMalloc(&rng_states, width * height * sizeof(curandState)));
            init_rng<<<gridSize, blockSize>>>(rng_states, width, height);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        }

        assert(sizeof(rgb) == pixel_stride);
        difference(src_buffer, width, height, src_stride, pixel_stride);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());


        uint8_t* dBuffer;
        size_t pitch;

        cudaError_t err;
        
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

	    // STEP2: Ouverture
	    uint8_t* eroded;
	    err = cudaMalloc(&eroded,width * sizeof(uint8_t) * height);
	    CHECK_CUDA_ERROR(err);
	    erosion_kernel<<<gridSize,blockSize>>>(dBuffer,eroded,width,height,pitch,pixel_stride,RADIUS);

	    cudaCheckError();

	    dilatation_kernel<<<gridSize,blockSize>>>(eroded,dBuffer,width,height,pitch,pixel_stride,RADIUS);

	    cudaCheckError();
	    cudaFree(eroded);

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
            err = cudaMemset(d_changed, changed_host, sizeof(bool));
            CHECK_CUDA_ERROR(err);

            hysteresis_propagation<<<gridSize, blockSize>>>(dBuffer, marker, candidate, width, height, pitch, pixel_stride, d_changed);
            cudaCheckError();

            err = cudaMemcpy(&changed_host, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
            CHECK_CUDA_ERROR(err);
        }
        //remove_red_channel_inp<<<gridSize, blockSize>>>(dBuffer, width, height, pitch);
        

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);
        cudaFree(marker);
        cudaFree(candidate);
        cudaFree(d_changed);
        
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);


    }
}
