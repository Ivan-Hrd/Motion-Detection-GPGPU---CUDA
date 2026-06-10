#include "filter_impl.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>
#include "logo.h"

struct rgb {
    uint8_t r, g, b;
};
#define K 7
#define RGB_DIFF_THRESHOLD 20
#define MAX_WEIGHTS 100
#define RADIUS 1

int res_width = 0;
int res_height = 0;

typedef struct {
        rgb rgbV;
        unsigned int w;
    }reservoir;
std::vector<std::array<reservoir, K>> rs;
std::vector<std::mt19937> rngs;

int find_matching_reservoir(const rgb& p, const std::vector<std::array<reservoir, K>>& rs, int idx) {
    int min_idx = -1;
    for (int i = 0; i < K; ++i) {
        if (rs[idx][i].w > 0) {
            if (abs((int)p.r - (int)rs[idx][i].rgbV.r) < RGB_DIFF_THRESHOLD &&
                abs((int)p.g - (int)rs[idx][i].rgbV.g) < RGB_DIFF_THRESHOLD &&
                abs((int)p.b - (int)rs[idx][i].rgbV.b) < RGB_DIFF_THRESHOLD) {
                min_idx = i;
                break;
            }
        }
        else {
            min_idx = i;
        }
    }
    return min_idx;
}

int min_res(const std::vector<std::array<reservoir, K>>& rs, int idx) {
    int min_idx = -1;
    int min_w = MAX_WEIGHTS + 1;
    for (int i = 0; i < K; ++i) {
        if (rs[idx][i].w < min_w) {
            min_idx = i;
            min_w = rs[idx][i].w;
        }
    }
    return min_idx;
}

int max_res(const std::vector<std::array<reservoir, K>>& rs, int idx) {
    int max_idx = -1;
    int max_w = -1;
    for (int i = 0; i < K; ++i) {
        if (rs[idx][i].w > max_w) {
            max_idx = i;
            max_w = rs[idx][i].w;
        }
    }
    return max_idx;
}

uint8_t get_gray_pixel(uint8_t* buffer, int x, int y, int stride)
{
    rgb* lineptr = (rgb*) (buffer + y * stride);
    return lineptr[x].r;
}

void erosion(uint8_t* buffer, std::vector<uint8_t>& eroded,
                int width, int height, int stride, int radius)
{
    eroded.resize(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t min_value = 255;

            for (int dy = -radius; dy <= radius; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= height) {
                    continue;
                }

                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= width) {
                        continue;
                    }

                    uint8_t value = get_gray_pixel(buffer, xx, yy, stride);
                    min_value = std::min(min_value, value);
                }
            }

            eroded[y * width + x] = min_value;
        }
    }
}

void dilatation(const std::vector<uint8_t>& input, uint8_t* buffer,
                 int width, int height, int stride, int radius)
{
    for (int y = 0; y < height; ++y) {
        rgb* lineptr = (rgb*) (buffer + y * stride);

        for (int x = 0; x < width; ++x) {
            uint8_t max_value = 0;

            for (int dy = -radius; dy <= radius; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= height) {
                    continue;
                }

                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= width) {
                        continue;
                    }

                    uint8_t value = input[yy * width + xx];
                    max_value = std::max(max_value, value);
                }
            }

            lineptr[x] = rgb{max_value, max_value, max_value};
        }
    }
}

extern "C" {

    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        // STEP 1 : difference
        if (res_height != height || res_width != width) {
            res_height = height;
            res_width = width;
            rs.clear();
            rs.resize(width*height);
            rngs.clear();
            rngs.resize(width*height);
            for (int i = 0; i < width*height; ++i) {
                for (int j = 0; j < K; ++j) {
                    rs[i][j] = {0};
                }
                rngs[i] = std::mt19937(i);
            }

        }
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                rgb p = lineptr[x];
                int idx = y * width + x;
                int m_idx = find_matching_reservoir(p, rs, idx);

                // random init
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float rand_val = dist(rngs[idx]);

                // update w and samples
                if (m_idx != -1 && rs[idx][m_idx].w > 0) { // match !
                    if (rs[idx][m_idx].w < MAX_WEIGHTS) {
                        rs[idx][m_idx].w++;
                        unsigned int tmpR1 = (unsigned int)rs[idx][m_idx].rgbV.r * (rs[idx][m_idx].w - 1) + (unsigned int)p.r;
                        unsigned int tmpG1 = (unsigned int)rs[idx][m_idx].rgbV.g * (rs[idx][m_idx].w - 1) + (unsigned int)p.g;
                        unsigned int tmpB1 = (unsigned int)rs[idx][m_idx].rgbV.b * (rs[idx][m_idx].w - 1) + (unsigned int)p.b;
                        uint8_t tmpR = (uint8_t)(tmpR1 / rs[idx][m_idx].w);
                        uint8_t tmpG = (uint8_t)(tmpG1 / rs[idx][m_idx].w);
                        uint8_t tmpB = (uint8_t)(tmpB1 / rs[idx][m_idx].w);
                        rs[idx][m_idx].rgbV = rgb{tmpR, tmpG, tmpB};
                    }
                    else {
                        rs[idx][m_idx].rgbV = p;
                    }
                }
                else if (m_idx != -1 && rs[idx][m_idx].w == 0) { // empty
                    rs[idx][m_idx].rgbV = p;
                    rs[idx][m_idx].w = 1;
                }
                else { // no match & no empty slot
                    int min_idx = min_res(rs, idx);
                    int total_weights = 0;
                    for (int i = 0; i < K; ++i) {
                        total_weights += rs[idx][i].w;
                    }
                    if (rand_val * total_weights >= rs[idx][min_idx].w) {
                        rs[idx][min_idx].rgbV = p;
                        rs[idx][min_idx].w = 1;
                    }
                }
                rgb background = rs[idx][max_res(rs, idx)].rgbV;
                int score = std::max(std::max(abs((int)p.r - (int)background.r),
                                      abs((int)p.g - (int)background.g)),
                                      abs((int)p.b - (int)background.b));
                lineptr[x] = rgb{(uint8_t)std::min(score, 255),
                                 (uint8_t)std::min(score, 255),
                                 (uint8_t)std::min(score, 255)};
            }
        }
        // STEP 2 : Ouverture
        static std::vector<uint8_t> eroded;

        erosion(buffer, eroded, width, height, stride, RADIUS);
        dilatation(eroded, buffer, width, height, stride, RADIUS);

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
