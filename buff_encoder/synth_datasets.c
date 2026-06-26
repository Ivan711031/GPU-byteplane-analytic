#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define DEV_COUNT 100000000ULL
#define FINAL_COUNT 1000000000ULL
#define DEFAULT_CHUNK_SIZE 1048576ULL
#define ZIPF_SUPPORT 8192U
#define ZIPF_LUT_SIZE 65536U
#define SENSOR_TARGET_INTERVAL 8192ULL

typedef enum dataset_kind {
    DATASET_SENSOR = 0,
    DATASET_UNIFORM = 1,
    DATASET_HEAVY_TAILED = 2,
    DATASET_ZIPFIAN = 3,
    DATASET_ALL = 4
} dataset_kind_t;

typedef struct rng_state {
    uint64_t state;
    bool has_spare_normal;
    double spare_normal;
} rng_state_t;

typedef struct stats {
    uint64_t count;
    long double mean;
    long double m2;
    double min;
    double max;
} stats_t;

typedef struct dataset_summary {
    dataset_kind_t kind;
    uint64_t count;
    uint64_t seed;
    uint64_t bytes_written;
    double min;
    double max;
    double mean;
    double stddev;
    char data_filename[PATH_MAX];
    char metadata_filename[PATH_MAX];
    char started_at[32];
    char finished_at[32];
} dataset_summary_t;

typedef struct zipf_table {
    uint32_t ranks[ZIPF_LUT_SIZE];
} zipf_table_t;

typedef struct sensor_state {
    double value;
    double target;
} sensor_state_t;

typedef struct generator_context {
    zipf_table_t zipf;
} generator_context_t;

static const char *dataset_slug(dataset_kind_t kind) {
    switch (kind) {
        case DATASET_SENSOR:
            return "sensor";
        case DATASET_UNIFORM:
            return "uniform";
        case DATASET_HEAVY_TAILED:
            return "heavy_tailed";
        case DATASET_ZIPFIAN:
            return "zipfian";
        default:
            return "all";
    }
}

static const char *dataset_title(dataset_kind_t kind) {
    switch (kind) {
        case DATASET_SENSOR:
            return "Sensor";
        case DATASET_UNIFORM:
            return "Uniform";
        case DATASET_HEAVY_TAILED:
            return "Heavy-tailed";
        case DATASET_ZIPFIAN:
            return "Zipfian";
        default:
            return "All";
    }
}

static const char *dataset_description(dataset_kind_t kind) {
    switch (kind) {
        case DATASET_SENSOR:
            return "Smooth temperature-like readings in roughly [15, 35] with low MSB entropy and narrow local FOR range.";
        case DATASET_UNIFORM:
            return "Uniform values in [0, 1000] so every mantissa bit carries signal.";
        case DATASET_HEAVY_TAILED:
            return "Log-normal samples with sigma=5 to span many orders of magnitude.";
        case DATASET_ZIPFIAN:
            return "Skewed Zipf-like values with rare large outliers to stress segment range expansion.";
        default:
            return "";
    }
}

static const char *dataset_distribution_json(dataset_kind_t kind) {
    switch (kind) {
        case DATASET_SENSOR:
            return "{\n"
                   "    \"family\": \"piecewise_seasonal_sensor\",\n"
                   "    \"value_range\": [15.0, 35.0],\n"
                    "    \"target_interval\": 8192,\n"
                   "    \"long_period\": 262144,\n"
                   "    \"mid_period\": 32768,\n"
                   "    \"target_noise_sigma\": 0.2,\n"
                   "    \"sensor_noise_sigma\": 0.018,\n"
                   "    \"mean_reversion\": 0.01\n"
                   "  }";
        case DATASET_UNIFORM:
            return "{\n"
                   "    \"family\": \"uniform\",\n"
                   "    \"value_range\": [0.0, 1000.0]\n"
                   "  }";
        case DATASET_HEAVY_TAILED:
            return "{\n"
                   "    \"family\": \"lognormal\",\n"
                   "    \"mu\": 0.0,\n"
                   "    \"sigma\": 5.0\n"
                   "  }";
        case DATASET_ZIPFIAN:
            return "{\n"
                   "    \"family\": \"truncated_zipf_with_outliers\",\n"
                   "    \"zipf_exponent\": 1.2,\n"
                   "    \"zipf_support\": 8192,\n"
                   "    \"outlier_probability\": 0.0001,\n"
                   "    \"outlier_scale\": 1000000.0,\n"
                   "    \"outlier_sigma\": 2.5\n"
                   "  }";
        default:
            return "{}";
    }
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--dataset sensor|uniform|heavy-tailed|zipfian|all]\n"
            "          [--profile dev|final] [--count N] [--out-dir DIR]\n"
            "          [--seed UINT64] [--chunk-size N]\n",
            argv0);
}

static bool parse_count_arg(const char *text, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long integer_value = strtoull(text, &end, 10);
    if (errno == 0 && end != NULL && *end == '\0') {
        *out = (uint64_t)integer_value;
        return true;
    }

    errno = 0;
    end = NULL;
    double floating_value = strtod(text, &end);
    if (errno != 0 || end == NULL || *end != '\0' || !isfinite(floating_value) || floating_value < 0.0) {
        return false;
    }

    double rounded = nearbyint(floating_value);
    if (fabs(floating_value - rounded) > 0.0) {
        return false;
    }

    if (rounded > (double)UINT64_MAX) {
        return false;
    }

    *out = (uint64_t)rounded;
    return true;
}

static dataset_kind_t parse_dataset_kind(const char *value) {
    if (strcmp(value, "sensor") == 0) {
        return DATASET_SENSOR;
    }
    if (strcmp(value, "uniform") == 0) {
        return DATASET_UNIFORM;
    }
    if (strcmp(value, "heavy-tailed") == 0 || strcmp(value, "heavy_tailed") == 0) {
        return DATASET_HEAVY_TAILED;
    }
    if (strcmp(value, "zipfian") == 0) {
        return DATASET_ZIPFIAN;
    }
    if (strcmp(value, "all") == 0) {
        return DATASET_ALL;
    }
    return (dataset_kind_t)-1;
}

static uint64_t splitmix64_next(rng_state_t *rng) {
    uint64_t z = (rng->state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31U);
}

static void rng_seed(rng_state_t *rng, uint64_t seed) {
    rng->state = seed;
    rng->has_spare_normal = false;
    rng->spare_normal = 0.0;
    (void)splitmix64_next(rng);
}

static double rng_uniform01(rng_state_t *rng) {
    const double scale = 1.0 / 9007199254740992.0;
    return (double)(splitmix64_next(rng) >> 11U) * scale;
}

static double rng_normal01(rng_state_t *rng) {
    if (rng->has_spare_normal) {
        rng->has_spare_normal = false;
        return rng->spare_normal;
    }

    double u1 = rng_uniform01(rng);
    double u2 = rng_uniform01(rng);
    if (u1 <= 0.0) {
        u1 = 0x1.0p-53;
    }

    double radius = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    rng->spare_normal = radius * sin(theta);
    rng->has_spare_normal = true;
    return radius * cos(theta);
}

static double clamp_double(double value, double low, double high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static void stats_init(stats_t *stats) {
    stats->count = 0;
    stats->mean = 0.0L;
    stats->m2 = 0.0L;
    stats->min = DBL_MAX;
    stats->max = -DBL_MAX;
}

static void stats_update(stats_t *stats, double value) {
    stats->count += 1;
    if (value < stats->min) {
        stats->min = value;
    }
    if (value > stats->max) {
        stats->max = value;
    }

    long double delta = (long double)value - stats->mean;
    stats->mean += delta / (long double)stats->count;
    long double delta2 = (long double)value - stats->mean;
    stats->m2 += delta * delta2;
}

static double stats_mean(const stats_t *stats) {
    return (double)stats->mean;
}

static double stats_stddev(const stats_t *stats) {
    if (stats->count == 0) {
        return 0.0;
    }
    return sqrt((double)(stats->m2 / (long double)stats->count));
}

static int ensure_directory(const char *path) {
    char partial[PATH_MAX];
    size_t length = strlen(path);
    if (length == 0 || length >= sizeof(partial)) {
        return -1;
    }

    memcpy(partial, path, length + 1);
    for (size_t i = 1; i < length; ++i) {
        if (partial[i] == '/') {
            partial[i] = '\0';
            if (mkdir(partial, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            partial[i] = '/';
        }
    }

    if (mkdir(partial, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static void format_timestamp_utc(time_t timestamp, char *buffer, size_t buffer_size) {
    struct tm tm_value;
    gmtime_r(&timestamp, &tm_value);
    strftime(buffer, buffer_size, "%Y-%m-%dT%H:%M:%SZ", &tm_value);
}

static void build_zipf_table(zipf_table_t *table) {
    static double cdf[ZIPF_SUPPORT];
    const double exponent = 1.2;
    double total = 0.0;
    for (uint32_t rank = 1; rank <= ZIPF_SUPPORT; ++rank) {
        total += 1.0 / pow((double)rank, exponent);
    }

    double running = 0.0;
    for (uint32_t rank = 1; rank <= ZIPF_SUPPORT; ++rank) {
        running += (1.0 / pow((double)rank, exponent)) / total;
        cdf[rank - 1] = running;
    }
    cdf[ZIPF_SUPPORT - 1] = 1.0;

    uint32_t rank_index = 0;
    for (uint32_t q = 0; q < ZIPF_LUT_SIZE; ++q) {
        double quantile = ((double)q + 0.5) / (double)ZIPF_LUT_SIZE;
        while (rank_index + 1 < ZIPF_SUPPORT && cdf[rank_index] < quantile) {
            rank_index += 1;
        }
        table->ranks[q] = rank_index + 1;
    }
}

static double sample_sensor(uint64_t index, rng_state_t *rng, sensor_state_t *sensor) {
    if (index % SENSOR_TARGET_INTERVAL == 0) {
        double phase_long = (2.0 * M_PI * (double)index) / 262144.0;
        double phase_mid = (2.0 * M_PI * (double)index) / 32768.0;
        double target = 25.0 + 6.0 * sin(phase_long) + 2.25 * sin(phase_mid) + 0.2 * rng_normal01(rng);
        sensor->target = clamp_double(target, 15.2, 34.8);
    }

    sensor->value += 0.01 * (sensor->target - sensor->value) + 0.018 * rng_normal01(rng);
    sensor->value = clamp_double(sensor->value, 15.0, 35.0);
    return sensor->value;
}

static double sample_uniform(rng_state_t *rng) {
    return 1000.0 * rng_uniform01(rng);
}

static double sample_heavy_tailed(rng_state_t *rng) {
    return exp(5.0 * rng_normal01(rng));
}

static double sample_zipfian(const zipf_table_t *table, rng_state_t *rng) {
    if (rng_uniform01(rng) < 0.0001) {
        return 1000000.0 * (1.0 + exp(2.5 * rng_normal01(rng)));
    }

    uint32_t rank = table->ranks[splitmix64_next(rng) >> 48U];
    return (double)rank + 0.001 * rng_uniform01(rng);
}

static double sample_value(dataset_kind_t kind, uint64_t index, generator_context_t *context, rng_state_t *rng,
                           sensor_state_t *sensor) {
    switch (kind) {
        case DATASET_SENSOR:
            return sample_sensor(index, rng, sensor);
        case DATASET_UNIFORM:
            return sample_uniform(rng);
        case DATASET_HEAVY_TAILED:
            return sample_heavy_tailed(rng);
        case DATASET_ZIPFIAN:
            return sample_zipfian(&context->zipf, rng);
        default:
            return 0.0;
    }
}

static int write_dataset_metadata(const char *metadata_path, dataset_kind_t kind, const dataset_summary_t *summary) {
    char temp_path[PATH_MAX];
    if (snprintf(temp_path, sizeof(temp_path), "%s.tmp", metadata_path) >= (int)sizeof(temp_path)) {
        fprintf(stderr, "Metadata path too long: %s\n", metadata_path);
        return -1;
    }

    FILE *file = fopen(temp_path, "wb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open metadata file %s: %s\n", temp_path, strerror(errno));
        return -1;
    }

    int ok = fprintf(file,
                     "{\n"
                     "  \"dataset\": \"%s\",\n"
                     "  \"slug\": \"%s\",\n"
                     "  \"description\": \"%s\",\n"
                     "  \"count\": %" PRIu64 ",\n"
                     "  \"dtype\": \"float64\",\n"
                     "  \"endianness\": \"little\",\n"
                     "  \"row_bytes\": 8,\n"
                     "  \"total_bytes\": %" PRIu64 ",\n"
                     "  \"seed\": %" PRIu64 ",\n"
                     "  \"data_file\": \"%s\",\n"
                     "  \"generated_at\": {\n"
                     "    \"started_utc\": \"%s\",\n"
                     "    \"finished_utc\": \"%s\"\n"
                     "  },\n"
                     "  \"distribution\": %s,\n"
                     "  \"summary\": {\n"
                     "    \"min\": %.17g,\n"
                     "    \"max\": %.17g,\n"
                     "    \"mean\": %.17g,\n"
                     "    \"stddev\": %.17g\n"
                     "  }\n"
                     "}\n",
                     dataset_title(kind),
                     dataset_slug(kind),
                     dataset_description(kind),
                     summary->count,
                     summary->bytes_written,
                     summary->seed,
                     summary->data_filename,
                     summary->started_at,
                     summary->finished_at,
                     dataset_distribution_json(kind),
                     summary->min,
                     summary->max,
                     summary->mean,
                     summary->stddev)
             >= 0;

    if (fclose(file) != 0) {
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "Failed to write metadata file %s\n", temp_path);
        remove(temp_path);
        return -1;
    }

    if (rename(temp_path, metadata_path) != 0) {
        fprintf(stderr, "Failed to rename metadata file %s -> %s: %s\n", temp_path, metadata_path, strerror(errno));
        remove(temp_path);
        return -1;
    }
    return 0;
}

static int write_manifest(const char *out_dir, const dataset_summary_t *summaries, size_t count) {
    char manifest_path[PATH_MAX];
    char temp_path[PATH_MAX];
    if (snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", out_dir) >= (int)sizeof(manifest_path)) {
        fprintf(stderr, "Manifest path too long for %s\n", out_dir);
        return -1;
    }
    if (snprintf(temp_path, sizeof(temp_path), "%s.tmp", manifest_path) >= (int)sizeof(temp_path)) {
        fprintf(stderr, "Manifest temp path too long for %s\n", manifest_path);
        return -1;
    }

    FILE *file = fopen(temp_path, "wb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open manifest %s: %s\n", temp_path, strerror(errno));
        return -1;
    }

    bool ok = true;
    if (fprintf(file, "{\n  \"datasets\": [\n") < 0) {
        ok = false;
    }

    for (size_t i = 0; ok && i < count; ++i) {
        const dataset_summary_t *summary = &summaries[i];
        if (fprintf(file,
                    "    {\n"
                    "      \"dataset\": \"%s\",\n"
                    "      \"slug\": \"%s\",\n"
                    "      \"count\": %" PRIu64 ",\n"
                    "      \"dtype\": \"float64\",\n"
                    "      \"data_file\": \"%s\",\n"
                    "      \"metadata_file\": \"%s\",\n"
                    "      \"total_bytes\": %" PRIu64 "\n"
                    "    }%s\n",
                    dataset_title(summary->kind),
                    dataset_slug(summary->kind),
                    summary->count,
                    summary->data_filename,
                    summary->metadata_filename,
                    summary->bytes_written,
                    (i + 1 == count) ? "" : ",")
                < 0) {
            ok = false;
        }
    }

    if (ok && fprintf(file, "  ]\n}\n") < 0) {
        ok = false;
    }

    if (fclose(file) != 0) {
        ok = false;
    }

    if (!ok) {
        fprintf(stderr, "Failed to write manifest %s\n", temp_path);
        remove(temp_path);
        return -1;
    }

    if (rename(temp_path, manifest_path) != 0) {
        fprintf(stderr, "Failed to rename manifest %s -> %s: %s\n", temp_path, manifest_path, strerror(errno));
        remove(temp_path);
        return -1;
    }

    return 0;
}

static int generate_one_dataset(dataset_kind_t kind, uint64_t count, uint64_t seed, uint64_t chunk_size,
                                const char *out_dir, generator_context_t *context, dataset_summary_t *summary) {
    char data_path[PATH_MAX];
    char metadata_path[PATH_MAX];
    char temp_data_path[PATH_MAX];
    const char *slug = dataset_slug(kind);

    if (snprintf(data_path, sizeof(data_path), "%s/%s.f64le.bin", out_dir, slug) >= (int)sizeof(data_path)) {
        fprintf(stderr, "Data path too long for %s\n", slug);
        return -1;
    }
    if (snprintf(metadata_path, sizeof(metadata_path), "%s/%s.meta.json", out_dir, slug) >= (int)sizeof(metadata_path)) {
        fprintf(stderr, "Metadata path too long for %s\n", slug);
        return -1;
    }
    if (snprintf(temp_data_path, sizeof(temp_data_path), "%s.tmp", data_path) >= (int)sizeof(temp_data_path)) {
        fprintf(stderr, "Temp data path too long for %s\n", slug);
        return -1;
    }

    FILE *data_file = fopen(temp_data_path, "wb");
    if (data_file == NULL) {
        fprintf(stderr, "Failed to open data file %s: %s\n", temp_data_path, strerror(errno));
        return -1;
    }

    double *buffer = malloc((size_t)chunk_size * sizeof(double));
    if (buffer == NULL) {
        fprintf(stderr, "Failed to allocate %" PRIu64 " doubles for chunk buffer\n", chunk_size);
        fclose(data_file);
        remove(temp_data_path);
        return -1;
    }

    rng_state_t rng;
    sensor_state_t sensor;
    stats_t stats;
    uint64_t dataset_seed = seed + (uint64_t)kind * UINT64_C(0x9e3779b97f4a7c15);
    rng_seed(&rng, dataset_seed);
    sensor.value = 24.0;
    sensor.target = 24.0;
    stats_init(&stats);

    time_t started = time(NULL);
    char started_at[32];
    char finished_at[32];
    format_timestamp_utc(started, started_at, sizeof(started_at));

    fprintf(stdout, "Generating %-12s -> %s (%" PRIu64 " rows)\n", dataset_title(kind), data_path, count);
    fflush(stdout);

    uint64_t generated = 0;
    uint64_t next_progress = count / 20;
    if (next_progress == 0) {
        next_progress = count;
    }

    while (generated < count) {
        uint64_t remaining = count - generated;
        uint64_t current_chunk = remaining < chunk_size ? remaining : chunk_size;
        for (uint64_t i = 0; i < current_chunk; ++i) {
            double value = sample_value(kind, generated + i, context, &rng, &sensor);
            buffer[i] = value;
            stats_update(&stats, value);
        }

        size_t written = fwrite(buffer, sizeof(double), (size_t)current_chunk, data_file);
        if (written != (size_t)current_chunk) {
            fprintf(stderr, "Short write for %s after %" PRIu64 " rows: %s\n", slug, generated, strerror(errno));
            free(buffer);
            fclose(data_file);
            remove(temp_data_path);
            return -1;
        }

        generated += current_chunk;
        if (generated == count || generated >= next_progress) {
            fprintf(stdout, "  %-12s progress: %6.2f%% (%" PRIu64 "/%" PRIu64 ")\n", dataset_title(kind),
                    100.0 * (double)generated / (double)count, generated, count);
            fflush(stdout);
            if (next_progress < count) {
                next_progress += count / 20 == 0 ? count : count / 20;
            }
        }
    }

    free(buffer);

    if (fclose(data_file) != 0) {
        fprintf(stderr, "Failed to close %s: %s\n", temp_data_path, strerror(errno));
        remove(temp_data_path);
        return -1;
    }

    if (rename(temp_data_path, data_path) != 0) {
        fprintf(stderr, "Failed to rename data file %s -> %s: %s\n", temp_data_path, data_path, strerror(errno));
        remove(temp_data_path);
        return -1;
    }

    time_t finished = time(NULL);
    format_timestamp_utc(finished, finished_at, sizeof(finished_at));

    memset(summary, 0, sizeof(*summary));
    summary->kind = kind;
    summary->count = count;
    summary->seed = dataset_seed;
    summary->bytes_written = count * sizeof(double);
    summary->min = stats.min;
    summary->max = stats.max;
    summary->mean = stats_mean(&stats);
    summary->stddev = stats_stddev(&stats);
    strncpy(summary->started_at, started_at, sizeof(summary->started_at) - 1);
    strncpy(summary->finished_at, finished_at, sizeof(summary->finished_at) - 1);
    const char *data_name = strrchr(data_path, '/');
    const char *metadata_name = strrchr(metadata_path, '/');
    data_name = data_name == NULL ? data_path : data_name + 1;
    metadata_name = metadata_name == NULL ? metadata_path : metadata_name + 1;
    strncpy(summary->data_filename, data_name, sizeof(summary->data_filename) - 1);
    strncpy(summary->metadata_filename, metadata_name, sizeof(summary->metadata_filename) - 1);

    if (write_dataset_metadata(metadata_path, kind, summary) != 0) {
        return -1;
    }

    fprintf(stdout,
            "Finished %-12s min=%.6g max=%.6g mean=%.6g std=%.6g bytes=%" PRIu64 "\n",
            dataset_title(kind),
            summary->min,
            summary->max,
            summary->mean,
            summary->stddev,
            summary->bytes_written);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    dataset_kind_t dataset = DATASET_ALL;
    uint64_t count = DEV_COUNT;
    uint64_t seed = 20260413ULL;
    uint64_t chunk_size = DEFAULT_CHUNK_SIZE;
    const char *out_dir = "data/dev";

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--dataset") == 0 && i + 1 < argc) {
            dataset = parse_dataset_kind(argv[++i]);
            if ((int)dataset < 0) {
                fprintf(stderr, "Unknown dataset kind: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(arg, "--profile") == 0 && i + 1 < argc) {
            const char *profile = argv[++i];
            if (strcmp(profile, "dev") == 0) {
                count = DEV_COUNT;
                out_dir = "data/dev";
            } else if (strcmp(profile, "final") == 0) {
                count = FINAL_COUNT;
                out_dir = "data/final";
            } else {
                fprintf(stderr, "Unknown profile: %s\n", profile);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(arg, "--count") == 0 && i + 1 < argc) {
            if (!parse_count_arg(argv[++i], &count) || count == 0) {
                fprintf(stderr, "Invalid count: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            if (!parse_count_arg(argv[++i], &seed)) {
                fprintf(stderr, "Invalid seed: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(arg, "--chunk-size") == 0 && i + 1 < argc) {
            if (!parse_count_arg(argv[++i], &chunk_size) || chunk_size == 0) {
                fprintf(stderr, "Invalid chunk size: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(arg, "--out-dir") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (ensure_directory(out_dir) != 0) {
        fprintf(stderr, "Failed to create output directory %s: %s\n", out_dir, strerror(errno));
        return 1;
    }

    generator_context_t context;
    build_zipf_table(&context.zipf);

    dataset_summary_t summaries[4];
    size_t summary_count = 0;

    if (dataset == DATASET_ALL) {
        dataset_kind_t kinds[] = {DATASET_SENSOR, DATASET_UNIFORM, DATASET_HEAVY_TAILED, DATASET_ZIPFIAN};
        for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
            if (generate_one_dataset(kinds[i], count, seed, chunk_size, out_dir, &context, &summaries[summary_count]) != 0) {
                return 1;
            }
            summary_count += 1;
        }
    } else {
        if (generate_one_dataset(dataset, count, seed, chunk_size, out_dir, &context, &summaries[0]) != 0) {
            return 1;
        }
        summary_count = 1;
    }

    if (write_manifest(out_dir, summaries, summary_count) != 0) {
        return 1;
    }

    fprintf(stdout, "Wrote manifest to %s/manifest.json\n", out_dir);
    return 0;
}
