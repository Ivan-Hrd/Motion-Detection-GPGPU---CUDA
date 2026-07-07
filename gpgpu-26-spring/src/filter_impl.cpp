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
#define K 10
#define RGB_DIFF_THRESHOLD 15
#define MAX_WEIGHTS 100
#define RADIUS 1
#define LOW 30
#define HIGH 40

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

uint8_t get_gray_pixel(uint8_t* buffer, int x, int y, int stride, int pixel_stride)
{
    uint8_t* pixel = buffer + y * stride + x * pixel_stride;
    return pixel[0];
}

void erosion(uint8_t* buffer, std::vector<uint8_t>& eroded,
                int width, int height, int stride, int pixel_stride, int radius)
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

                    uint8_t value = get_gray_pixel(buffer, xx, yy, stride, pixel_stride);
                    min_value = std::min(min_value, value);
                }
            }

            eroded[y * width + x] = min_value;
        }
    }
}

void dilatation(const std::vector<uint8_t>& input, uint8_t* buffer,
                 int width, int height, int stride, int pixel_stride, int radius)
{
    for (int y = 0; y < height; ++y) {
        uint8_t* lineptr = buffer + y * stride;

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

            lineptr[x * pixel_stride] = max_value;
            lineptr[x * pixel_stride + 1] = max_value;
            lineptr[x * pixel_stride + 2] = max_value;
        }
    }
}

void hysteresis(uint8_t* buffer, int width, int height, int stride, int pixel_stride, int low, int high)
{
    static std::vector<bool> marker;
    marker.assign(width * height, false);

    static std::vector<bool> candidate;
    candidate.assign(width * height, false);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            int i = y * width + x;
            uint8_t value = buffer[y * stride + x * pixel_stride];

            candidate[i] = value >= low;
            marker[i] = value >= high;

            buffer[y * stride + x * pixel_stride] = 0;
            buffer[y * stride + x * pixel_stride + 1] = 0;
            buffer[y * stride + x * pixel_stride + 2] = 0;
        }
    }

    bool changed = true;

    while (changed)
    {
        changed = false;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                int i = y * width + x;

                if (buffer[y * stride + x * pixel_stride]) continue;

                if (!candidate[i]) continue;

                if (marker[i])
                {
                    buffer[y * stride + x * pixel_stride] = 255;
                    buffer[y * stride + x * pixel_stride + 1] = 255;
                    buffer[y * stride + x * pixel_stride + 2] = 255;

                    changed = true;
                    continue;
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

                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}

void difference(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
{
    // Initialisation si les dimensions changent
    if (res_height != height || res_width != width) {
        res_height = height;
        res_width = width;
        rs.clear();
        rs.resize(width * height);
        rngs.clear();
        rngs.resize(width * height);
        for (int i = 0; i < width * height; ++i) {
            for (int j = 0; j < K; ++j) {
                rs[i][j] = {0};
            }
            rngs[i] = std::mt19937(i);
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t* lineptr = buffer + y * stride + x * pixel_stride;
            rgb p = rgb{
                lineptr[0],
                lineptr[1],
                lineptr[2]
            };

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
                    rs[idx][m_idx].rgbV.r = (uint8_t)(tmpR1 / rs[idx][m_idx].w);
                    rs[idx][m_idx].rgbV.g = (uint8_t)(tmpG1 / rs[idx][m_idx].w);
                    rs[idx][m_idx].rgbV.b = (uint8_t)(tmpB1 / rs[idx][m_idx].w);
                } else {
                    // Keep updating with moving average even when max weight reached
                    unsigned int tmpR1 = (unsigned int)rs[idx][m_idx].rgbV.r * (MAX_WEIGHTS - 1) + (unsigned int)p.r;
                    unsigned int tmpG1 = (unsigned int)rs[idx][m_idx].rgbV.g * (MAX_WEIGHTS - 1) + (unsigned int)p.g;
                    unsigned int tmpB1 = (unsigned int)rs[idx][m_idx].rgbV.b * (MAX_WEIGHTS - 1) + (unsigned int)p.b;
                    rs[idx][m_idx].rgbV.r = (uint8_t)(tmpR1 / MAX_WEIGHTS);
                    rs[idx][m_idx].rgbV.g = (uint8_t)(tmpG1 / MAX_WEIGHTS);
                    rs[idx][m_idx].rgbV.b = (uint8_t)(tmpB1 / MAX_WEIGHTS);
                }
            } else if (m_idx != -1 && rs[idx][m_idx].w == 0) { // empty
                rs[idx][m_idx].rgbV = p;
                rs[idx][m_idx].w = 1;
            } else { // no match & no empty slot
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



#define BG_MIN_WEIGHT 30
            int score = 255;
            bool found_established = false;

            for (int i = 0; i < K; ++i) {
                if (rs[idx][i].w >= BG_MIN_WEIGHT) {
                    found_established = true;
                    int d = std::max(
                        abs((int)p.r - (int)rs[idx][i].rgbV.r),
                        std::max(abs((int)p.g - (int)rs[idx][i].rgbV.g),
                        abs((int)p.b - (int)rs[idx][i].rgbV.b))
                    );
                    score = std::min(score, d);
                }
            }

            // Si aucun n'est assez lourd (poids < 30),
            // on utilise le plus lourd par défaut pour ne pas rester bloqué
            if (!found_established) {
                rgb background = rs[idx][max_res(rs, idx)].rgbV;
                score = std::max(std::max(abs((int)p.r - (int)background.r),
                                      abs((int)p.g - (int)background.g)),
                                      abs((int)p.b - (int)background.b));
            }
            uint8_t value = 0;
            value = static_cast<uint8_t>(std::min(score, 255));
            

            lineptr[0] = value;
            lineptr[1] = value;
            lineptr[2] = value;
        }
    }
}



uint8_t* copie_input(uint8_t* input,int width, int height,int stride,int pixel_stride)
{
    uint8_t * copie = new uint8_t[stride * height];
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            uint8_t* lineptr = input + y * stride + x * pixel_stride;
            uint8_t* copieptr = copie + y * stride + x * pixel_stride;
            copieptr[0]  = lineptr[0];
            copieptr[1]  = lineptr[1];
            copieptr[2]  = lineptr[2];
            if (pixel_stride == 4)
            {
                copieptr[3] = lineptr[3];
            }
        }
    }
    return copie;
}
void masquage(uint8_t* input,uint8_t*mask, int width, int height,int stride, int pixel_stride)
{
    rgb red = {255,0,0};
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            uint8_t* lineptr = input + y * stride + x * pixel_stride;
            uint8_t* maskptr = mask + y * stride + x * pixel_stride;
            
            // partie rouge mis entre 0 et 1 (facteur)
            float m = maskptr[0] / 255.0f;
            
            // input = input + 0.5 * red * masque
            lineptr[0] = static_cast<uint8_t>(std::min(255.0f,lineptr[0] + 0.5f * red.r * m));
        }
    }
}


extern "C" {

    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {

        // STEP 0: copie input
        uint8_t* mask = copie_input(buffer,width, height,stride,pixel_stride);

        // STEP 1 : difference
        difference(mask, width, height, stride, pixel_stride);


        // STEP 2 : Ouverture
        static std::vector<uint8_t> eroded;

        erosion(mask, eroded, width, height, stride, pixel_stride, RADIUS);
        dilatation(eroded, mask, width, height, stride, pixel_stride, RADIUS);

        // STEP 3: Seuillage d’hystérésis
        hysteresis(mask, width, height, stride, pixel_stride, LOW, HIGH);

        // STEP 4: masquage
        masquage(buffer,mask,width,height,stride,pixel_stride);
        delete[] mask;
        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
