#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#ifndef F_PI
#define F_PI (float)M_PI
#endif

// print debugging messages?
#ifndef DEBUG
#define DEBUG false
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES 10
#endif

// ranges for the random numbers:
const float GMIN = 10.0;   // ground distance in meters
const float GMAX = 20.0;   // ground distance in meters
const float HMIN = 20.0;   // cliff height in meters
const float HMAX = 30.0;   // cliff height in meters
const float DMIN = 10.0;   // distance to castle in meters
const float DMAX = 20.0;   // distance to castle in meters
const float VMIN = 20.0;   // initial cannonball velocity in m/s
const float VMAX = 30.0;   // initial cannonball velocity in m/s
const float THMIN = 70.0;  // cannonball launch angle in degrees
const float THMAX = 80.0;  // cannonball launch angle in degrees

const float GRAVITY = -9.8;    // acceleration due to gravity in m/s^2
const float TOL = 5.0;         // tolerance in meters for a hit

// function prototypes:
float Ranf(float, float);
void TimeOfDaySeed();

// degrees-to-radians:
inline float Radians(float degrees) {
    return (F_PI / 180.f) * degrees;
}

// random float in range:
float Ranf(float low, float high) {
    float r = (float)rand();
    float t = r / (float)RAND_MAX;
    return low + t * (high - low);
}

// call this to use different random seed each run:
void TimeOfDaySeed() {
    time_t now;
    time(&now);
    struct tm *ptm = localtime(&now);
    srand(1000 * (60 * ptm->tm_min + ptm->tm_sec));
}

// main program:
int main() {
#ifndef _OPENMP
    fprintf(stderr, "No OpenMP support!\n");
    return 1;
#endif

    TimeOfDaySeed();

    int threadCounts[] = {1, 2, 4, 6, 8};
    int trialCounts[] = {100, 1000, 10000, 50000, 100000};

    // CSV header for Excel import:
    printf("Threads,Trials,MegaTrialsPerSec\n");

    // loop through each thread and trial combination:
    for (int t = 0; t < sizeof(threadCounts)/sizeof(int); t++) {
        for (int tr = 0; tr < sizeof(trialCounts)/sizeof(int); tr++) {
            int NUMT = threadCounts[t];
            int NUMTRIALS = trialCounts[tr];
            omp_set_num_threads(NUMT);

            float *vs = new float[NUMTRIALS];
            float *ths = new float[NUMTRIALS];
            float *gs = new float[NUMTRIALS];
            float *hs = new float[NUMTRIALS];
            float *ds = new float[NUMTRIALS];

            // fill arrays with randomized inputs:
            for (int i = 0; i < NUMTRIALS; i++) {
                vs[i] = Ranf(VMIN, VMAX);
                ths[i] = Ranf(THMIN, THMAX);
                gs[i] = Ranf(GMIN, GMAX);
                hs[i] = Ranf(HMIN, HMAX);
                ds[i] = Ranf(DMIN, DMAX);
            }

            double maxPerformance = 0.;
            int numHits = 0;

            // NUMTRIES for max performance
            for (int tries = 0; tries < NUMTRIES; tries++) {
                double time0 = omp_get_wtime();
                int hits = 0;

                #pragma omp parallel for default(none) shared(vs, ths, gs, hs, ds, NUMTRIALS) reduction(+:hits)
                for (int i = 0; i < NUMTRIALS; i++) {
                    float v = vs[i];
                    float thr = Radians(ths[i]);
                    float vx = v * cos(thr);
                    float vy = v * sin(thr);
                    float g = gs[i];
                    float h = hs[i];
                    float d = ds[i];

                    float t_flight = (-2. * vy) / GRAVITY;
                    float x = vx * t_flight;
                    if (x <= g) continue; // didn't reach cliff

                    float t_cliff = g / vx;
                    float y = vy * t_cliff + 0.5 * GRAVITY * t_cliff * t_cliff;
                    if (y <= h) continue; // hit cliff face

                    float A = 0.5 * GRAVITY;
                    float B = vy;
                    float C = -h;
                    float discriminant = B * B - 4 * A * C;
                    if (discriminant < 0) continue; // shouldn't happen

                    float sqrtDisc = sqrtf(discriminant);
                    float t1 = (-B + sqrtDisc) / (2. * A);
                    float t2 = (-B - sqrtDisc) / (2. * A);
                    float t_upper = t1 > t2 ? t1 : t2;

                    float x_land = vx * t_upper - g;
                    if (fabs(x_land - d) <= TOL) hits++;
                }

                double time1 = omp_get_wtime();
                double megaTrialsPerSecond = (double)NUMTRIALS / (time1 - time0) / 1e6;
                if (megaTrialsPerSecond > maxPerformance)
                    maxPerformance = megaTrialsPerSecond;

                numHits = hits;
            }

            float probability = (float)numHits / (float)(NUMTRIALS);

            // output for Excel
            printf("%d,%d,%6.2lf\n", NUMT, NUMTRIALS, maxPerformance);

            delete[] vs;
            delete[] ths;
            delete[] gs;
            delete[] hs;
            delete[] ds;
        }
    }

    return 0;
}
