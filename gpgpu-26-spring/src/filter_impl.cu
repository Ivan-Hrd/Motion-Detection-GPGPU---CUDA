#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include "curand_kernel.h"
#include "filter_impl.h"
#include "logo.h"

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
uint8_t* dBuffer = nullptr;
bool* marker = nullptr;
bool* candidate = nullptr;
int* d_changed = nullptr;
uint8_t* dOriginal = nullptr;
uint8_t* eroded = nullptr;

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

__device__ int matching_reservoir(rgb p, reservoir* res, int width, int height)
{
    int empty = -1;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size = width * height;
    int idx = y * width + x;

    for (int j = 0; j < K; j++)
    {
        reservoir r = res[j*size+idx];
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


__global__ void difference_kernel(uint8_t* buffer, reservoir* reservoirs, curandState* states,
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

    int m_idx = matching_reservoir(p, reservoirs, width, height);

    float rand_val = curand_uniform(&states[idx]);

    int global_idx = m_idx*height*width+idx;
    if (m_idx != -1 && reservoirs[global_idx].w > 0)
    {
        unsigned int w = reservoirs[global_idx].w;
        if (w < MAX_WEIGHTS)
        {
            reservoirs[global_idx].w++;
            w = reservoirs[global_idx].w;
            reservoirs[global_idx].rgbV.r = (uint8_t)(((unsigned int)reservoirs[global_idx].rgbV.r * (w - 1) + p.r) / w);
            reservoirs[global_idx].rgbV.g = (uint8_t)(((unsigned int)reservoirs[global_idx].rgbV.g * (w - 1) + p.g) / w);
            reservoirs[global_idx].rgbV.b = (uint8_t)(((unsigned int)reservoirs[global_idx].rgbV.b * (w - 1) + p.b) / w);
        }
        else
        {
            reservoirs[global_idx].rgbV.r = (uint8_t)(((unsigned int)reservoirs[global_idx].rgbV.r * (MAX_WEIGHTS - 1) + p.r) / MAX_WEIGHTS);
            reservoirs[global_idx].rgbV.g = (uint8_t)(((unsigned int)reservoirs[global_idx].rgbV.g * (MAX_WEIGHTS - 1) + p.g) / MAX_WEIGHTS);
            reservoirs[global_idx].rgbV.b = (uint8_t)(((unsigned int)reservoirs[global_idx].rgbV.b * (MAX_WEIGHTS - 1) + p.b) / MAX_WEIGHTS);
        }

        line_ptr[0] = 0;
        line_ptr[1] = 0;
        line_ptr[2] = 0;
    }
    else if (m_idx != -1 && reservoirs[global_idx].w == 0)
    {
        reservoirs[global_idx].rgbV = p;
        reservoirs[global_idx].w = 1;
    }
    else // Cas 3 : aucune correspondance, aucun slot vide
    {
        int min_idx = 0;
        for (int i = 1; i < K; i++)
            if (reservoirs[height*width*i+idx].w < reservoirs[height*width*min_idx+idx].w)
                min_idx = i;
        unsigned int total_w = 0;
        for (int i = 0; i < K; i++)
            total_w += reservoirs[height*width*i+idx].w;

        if (rand_val * total_w >= reservoirs[height*width*min_idx+idx].w)
        {
            reservoirs[height*width*min_idx+idx].rgbV = p;
            reservoirs[height*width*min_idx+idx].w = 1;
        }
    }

#define BG_MIN_WEIGHT 30
    int score = 255;
    bool found_established = false;

    for (int i = 0; i < K; i++)
    {
        if (reservoirs[height*width*i+idx].w >= BG_MIN_WEIGHT)
        {
            found_established = true;
            int d = max(
                abs((int)p.r - (int)reservoirs[height*width*i+idx].rgbV.r),
                max(
                    abs((int)p.g - (int)reservoirs[height*width*i+idx].rgbV.g),
                    abs((int)p.b - (int)reservoirs[height*width*i+idx].rgbV.b)
                )
            );
            score = min(score, d);
        }
    }

    if (!found_established)
    {
        int max_idx = 0;
        for (int i = 1; i < K; i++)
            if (reservoirs[height*width*i+idx].w > reservoirs[height*width*max_idx+idx].w)
                max_idx = i;

        rgb bg = reservoirs[height*width*max_idx+idx].rgbV;
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
                int pixel_stride)
{
    dim3 blockSize(32, 32);
    dim3 gridSize((width + 31)/32, (height + 31)/32);

    difference_kernel <<<gridSize, blockSize>>> (buffer, rs, rng_states, width, height, stride, pixel_stride);
}

void cleanup()
{
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
    if (eroded != nullptr) {
        cudaFree(eroded);
        eroded = nullptr;
    }
    if (dOriginal != nullptr) {
        cudaFree(dOriginal);
        dOriginal = nullptr;
    }
}


extern "C" 
{
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        static bool registered = false;
        assert(sizeof(rgb) == pixel_stride);
        static size_t pitch;
        static size_t original_pitch;

        cudaError_t err;
        if (!registered)
        {
            atexit(cleanup);
            registered = true;
            err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
            CHECK_CUDA_ERROR(err);
            CHECK_CUDA_ERROR(cudaMalloc(&marker, width * height * sizeof(bool)));
            CHECK_CUDA_ERROR(cudaMalloc(&candidate, width * height * sizeof(bool)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_changed, sizeof(int)));
            err = cudaMalloc(&eroded,width * sizeof(uint8_t) * height);
            CHECK_CUDA_ERROR(err);
            err = cudaMallocPitch(&dOriginal, &original_pitch, width * sizeof(rgb), height);
            CHECK_CUDA_ERROR(err);

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

            CHECK_CUDA_ERROR(
                cudaMalloc(&rs, width * height * K * sizeof(reservoir)));
            CHECK_CUDA_ERROR(
                cudaMemset(rs, 0, width * height * K * sizeof(reservoir)));
            CHECK_CUDA_ERROR(cudaMalloc(&rng_states, width * height * sizeof(curandState)));
            init_rng<<<gridSize, blockSize>>>(rng_states, width, height);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        err = cudaMemcpy2D(dOriginal, original_pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dBuffer, pitch, dOriginal, original_pitch, width * sizeof(rgb), height, cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERROR(err);

        assert(sizeof(rgb) == pixel_stride);
        difference(dBuffer, width, height, pitch, pixel_stride);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	    // STEP2: Ouverture
	    erosion_kernel<<<gridSize,blockSize>>>(dBuffer,eroded,width,height,pitch,pixel_stride,RADIUS);

	    cudaCheckError();

	    dilatation_kernel<<<gridSize,blockSize>>>(eroded,dBuffer,width,height,pitch,pixel_stride,RADIUS);

	    cudaCheckError();

        // STEP 3 : Hysteresis
        hysteresis_init<<<gridSize, blockSize>>>(dBuffer, marker, candidate, width, height, pitch, pixel_stride);
        cudaDeviceSynchronize();
        cudaCheckError();

        int h_count = 1;
        while (h_count > 0)
        {
            CHECK_CUDA_ERROR(cudaMemset(d_changed, 0, sizeof(int)));

            hysteresis_propagation<<<gridSize, blockSize>>>(dBuffer, marker, candidate,
                width, height, pitch, pixel_stride, d_changed);
            cudaCheckError();

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(&h_count, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA_ERROR(err);
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