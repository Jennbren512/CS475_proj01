#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <fstream>

#ifndef F_PI
#define F_PI (float)M_PI
#endif

const float GMIN = 10.0, GMAX = 20.0;
const float HMIN = 20.0, HMAX = 30.0;
const float DMIN = 10.0, DMAX = 20.0;
const float VMIN = 20.0, VMAX = 30.0;
const float THMIN = 70.0, THMAX = 80.0;
const float GRAVITY = -9.8;
const float TOL = 5.0;

const int NUMTRIES = 50;

float Ranf(float low, float high)
{
    float r = (float)rand();
    float t = r / (float)RAND_MAX;
    return low + t * (high - low);
}

int Ranf(int ilow, int ihigh)
{
    float low = (float)ilow;
    float high = ceil((float)ihigh);
    return (int)Ranf(low, high);
}

void TimeOfDaySeed()
{
    time_t now;
    time(&now);
    struct tm n = *localtime(&now);
    struct tm jan01 = *localtime(&now);
    jan01.tm_mon = 0;
    jan01.tm_mday = 1;
    jan01.tm_hour = 0;
    jan01.tm_min = 0;
    jan01.tm_sec = 0;
    double seconds = difftime(now, mktime(&jan01));
    unsigned int seed = (unsigned int)(1000. * seconds);
    srand(seed);
}

inline float Radians(float degrees)
{
    return (F_PI / 180.f) * degrees;
}

int main()
{
#ifndef _OPENMP
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

    TimeOfDaySeed();

    // Final thread and trial configs:
    int threadCounts[] = {1, 2, 4, 6, 8};
    int trialCounts[] = {1000, 10000, 50000, 100000, 200000, 300000, 400000, 500000};

    std::ofstream csvFile("simulation_results.csv");
    csvFile << "Threads,Trials,MegaTrialsPerSecond\n";

    // Optional: anchor chart with dummy zero points
    for (int i = 0; i < sizeof(threadCounts) / sizeof(int); i++)
        csvFile << threadCounts[i] << ",0,0.0\n";

    for (int t = 0; t < sizeof(threadCounts) / sizeof(int); t++)
    {
        int numThreads = threadCounts[t];
        omp_set_num_threads(numThreads);

        for (int tr = 0; tr < sizeof(trialCounts) / sizeof(int); tr++)
        {
            int NUMTRIALS = trialCounts[tr];

            float* vs = new float[NUMTRIALS];
            float* ths = new float[NUMTRIALS];
            float* gs = new float[NUMTRIALS];
            float* hs = new float[NUMTRIALS];
            float* ds = new float[NUMTRIALS];

            for (int n = 0; n < NUMTRIALS; n++)
            {
                vs[n] = Ranf(VMIN, VMAX);
                ths[n] = Ranf(THMIN, THMAX);
                gs[n] = Ranf(GMIN, GMAX);
                hs[n] = Ranf(HMIN, HMAX);
                ds[n] = Ranf(DMIN, DMAX);
            }

            double maxPerformance = 0.;
            int numHits = 0;

            for (int tries = 0; tries < NUMTRIES; tries++)
            {
                double time0 = omp_get_wtime();
                int hits = 0;

                #pragma omp parallel for reduction(+:hits)
                for (int n = 0; n < NUMTRIALS; n++)
                {
                    float v = vs[n];
                    float thr = Radians(ths[n]);
                    float vx = v * cos(thr);
                    float vy = v * sin(thr);
                    float g = gs[n];
                    float h = hs[n];
                    float d = ds[n];

                    float t_flight = -vy / (0.5 * GRAVITY);
                    float x = vx * t_flight;
                    if (x <= g) continue;

                    float t_cliff = g / vx;
                    float y_cliff = vy * t_cliff + 0.5 * GRAVITY * t_cliff * t_cliff;
                    if (y_cliff <= h) continue;

                    float A = 0.5 * GRAVITY;
                    float B = vy;
                    float C = -h;
                    float disc = B * B - 4.f * A * C;
                    if (disc < 0.) continue;

                    float t1 = (-B + sqrtf(disc)) / (2.f * A);
                    float t2 = (-B - sqrtf(disc)) / (2.f * A);
                    float tmax = fmaxf(t1, t2);
                    float upperDist = vx * tmax - g;

                    if (fabs(upperDist - d) <= TOL)
                        hits++;
                }

                double time1 = omp_get_wtime();
                double megaTrialsPerSecond = (double)NUMTRIALS / (time1 - time0) / 1e6;
                if (megaTrialsPerSecond > maxPerformance)
                {
                    maxPerformance = megaTrialsPerSecond;
                    numHits = hits;
                }
            }

            float probability = (float)numHits / (float)(NUMTRIALS);
            fprintf(stderr, "%2d threads : %8d trials ; probability = %6.2f%% ; megatrials/sec = %6.2lf\n",
                numThreads, NUMTRIALS, 100. * probability, maxPerformance);

            csvFile << numThreads << "," << NUMTRIALS << "," << maxPerformance << "\n";

            delete[] vs;
            delete[] ths;
            delete[] gs;
            delete[] hs;
            delete[] ds;
        }
    }

    csvFile.close();
    return 0;
}
