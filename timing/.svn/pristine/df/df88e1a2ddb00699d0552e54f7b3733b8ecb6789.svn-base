#include <cstdint>
#include <iostream>
#include <vector>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <perfmon/pfmlib.h>
#include "perf.h"

using namespace std;

/**
 * Define the events that are about to be measured
 */
string pfm_events_str [] = {
    "CPU_CLK_UNHALTED" //,
    //"PERF_COUNT_HW_INSTRUCTIONS"
};

/**
 * Then pre-compute the events count
 */
#define perf_events_count (sizeof(pfm_events_str) / sizeof(string))
std::vector<bool> cpu_cores;

perf_event_attr  perf_events [perf_events_count];
pfm_event_info_t pfm_events  [perf_events_count];
int perf_event_fd;

typedef struct {
    uint64_t nr;                    /* The number of events */
    struct {
        uint64_t value;             /* The value of the event */
        uint64_t id;                /* if PERF_FORMAT_ID */
    } values[perf_events_count];
} perf_read_format_t;
perf_read_format_t perf_event_values;


int get_highest_core ()
{
    int highest_core = 0;

    FILE * f_presentcpus = fopen("/sys/devices/system/cpu/present", "r");
    if (!f_presentcpus) {
        std::cerr << "Can not open /sys/devices/system/cpu/present file." << std::endl;
        exit(1);
    }

    char buffer[1024];
    if(NULL == fgets(buffer, 1024, f_presentcpus)) {
        std::cerr << "Can not read /sys/devices/system/cpu/present." << std::endl;
        exit(1);
    }
    fclose(f_presentcpus);

    unsigned int num_cores = 0;
    sscanf(buffer, "0-%d", &num_cores);
    if (num_cores == 0) {
        sscanf(buffer, "%d", &num_cores);
    }

    if (num_cores == 0) {
        std::cerr << "Can not read number of present cores" << std::endl;
        exit(1);
    } else {
        num_cores += 1;
        cpu_cores.resize(num_cores, false);
    }

    FILE * f_cpuinfo = fopen("/proc/cpuinfo", "r");
    if (!f_cpuinfo) {
        std::cerr << "Can not open /proc/cpuinfo file." << std::endl;
        exit(1);
    }

    while (0 != fgets(buffer, 1024, f_cpuinfo)) {
        if (strncmp(buffer, "processor", sizeof("processor") - 1) == 0) {
            int core_id;
            sscanf(buffer, "processor\t: %d", &core_id);
            if (core_id > highest_core) {
                highest_core = core_id;
            }
            cpu_cores[core_id] = true;
        }
    }
    fclose(f_cpuinfo);

    for (int i = 0; i < num_cores; i += 1) {
        cout << "Core " << i << "\t: ";
        if (cpu_cores[i]) {
            cout << "online" << endl;
        } else {
            cout << "offline" << endl;
        }
    }
    cout << "Scheduling on core: " << highest_core << endl << endl;
    return highest_core;
}


void pfm_init_events ()
{
    int i;
    int total_supported_events = 0;
    int total_available_events = 0;
    pfm_err_t pfm_code;

    pfm_code = pfm_initialize ();
    if (pfm_code != PFM_SUCCESS) {
        cerr << "libpfm4: initialization failed!" << endl;
        cerr << pfm_strerror(pfm_code) << endl;
        exit(1);
    } else {
        cout << "using libpfm4: " << pfm_get_version() << endl;
    }

    printf("Detected PMU models:\n");
    for (i = 0; i < PFM_PMU_MAX; i++) {
        pfm_pmu_info_t pinfo;
        memset(&pinfo, 0, sizeof(pfm_pmu_info_t));
        pfm_code = pfm_get_pmu_info((pfm_pmu_t) i, &pinfo);
        if (pfm_code != PFM_SUCCESS)
            continue;
        if (pinfo.is_present) {
            printf("\t[%d, %s, \"%s\"]\n", i, pinfo.name, pinfo.desc);
            total_supported_events += pinfo.nevents;
        }
        total_available_events += pinfo.nevents;
    }
    printf("Total events: %d available, %d supported\n\n", total_available_events, total_supported_events);

    for (i = 0; i < perf_events_count; i += 1)
    {
        // pfm_code = pfm_find_event(pfm_events_str[i].c_str());
        char *fqstr;
        pfm_pmu_encode_arg_t e;
        memset(&e, 0, sizeof(pfm_pmu_encode_arg_t));
        fqstr = NULL; e.fstr = &fqstr;
        pfm_code = pfm_get_os_event_encoding(pfm_events_str[i].c_str(), PFM_PLM0|PFM_PLM3, PFM_OS_NONE, &e);
        if (pfm_code != PFM_SUCCESS) {
            cerr << "Event " << pfm_events_str[i] << " failed!" << endl;
            cerr << pfm_strerror(pfm_code) << endl;
            exit(1);
        } else {
            free(fqstr);
        }

        int idx = e.idx;
        memset(&(pfm_events[i]), 0, sizeof(pfm_event_info_t));
        pfm_code = pfm_get_event_info(idx, PFM_OS_PERF_EVENT, &(pfm_events[i]));
        if (pfm_code < 0) {
            cerr << "Event " << pfm_events_str[i] << " failed!" << endl;
            cerr << pfm_strerror(pfm_code) << endl;
            exit(1);
        }

        pfm_pmu_info_t pmu_info;
        memset(&pmu_info, 0, sizeof(pfm_pmu_info_t));
        pfm_code = pfm_get_pmu_info(pfm_events[i].pmu, &pmu_info);
        if (pfm_code < 0) {
            cerr << "Event " << pfm_events_str[i] << " failed!" << endl;
            cerr << pfm_strerror(pfm_code) << endl;
            exit(1);
        }

        cout << "Name        : " << pfm_events[i].name << " (" << pmu_info.desc << ")" << endl;
        cout << "Description : " << pfm_events[i].desc << endl;
        cout << "Code        : 0x" << hex << pfm_events[i].code << dec << endl;
        cout << endl;
    }
}


void perf_init ()
{
    cout << "==============================================================" << endl;
    cout << "= Linux Perf Initializing" << endl;
    cout << "==============================================================" << endl;

    // schedule the process on the highest core
    int cpu = get_highest_core ();
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);
    sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpu_set);

    // Adjust the events on the architecture
    pfm_init_events();

    // initialize Linux Perf
    perf_event_fd = -1;

    for (int i = 0; i < perf_events_count; i += 1) {

        memset(&(perf_events[i]), 0, sizeof(struct perf_event_attr));

        // Get the PMU
        pfm_pmu_info_t pmu_info;
        memset(&pmu_info, 0, sizeof(pfm_pmu_info_t));

        perf_type_id type_id;
        if (pmu_info.pmu == PFM_PMU_PERF_EVENT) {
            type_id = PERF_TYPE_HARDWARE;
        } else {
            type_id = PERF_TYPE_RAW;
        }

        // Setup the damn thing
        perf_events[i].type = type_id; // PERF_TYPE_RAW; // PERF_TYPE_HARDWARE; // PERF_TYPE_RAW;
        perf_events[i].size = sizeof(struct perf_event_attr);
        perf_events[i].config = pfm_events[i].code; // PERF_COUNT_HW_CPU_CYCLES; // events[i].config;
        perf_events[i].disabled = 1;
        perf_events[i].exclude_kernel = 1;
        perf_events[i].exclude_hv = 1;
        perf_events[i].read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;

        perf_event_fd = (int) syscall(__NR_perf_event_open, &(perf_events[i]), 0, cpu, perf_event_fd, 0);
        if (perf_event_fd == -1) {
            cerr << "Linux Perf initialization failed: " << strerror(errno) << endl;
            exit(1);
        }
    }

    cout << "==============================================================" << endl;
    cout << "= Using Linux Perf" << endl;
    cout << "==============================================================" << endl;
}

void cycles_count_start ()
{
    ioctl(perf_event_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(perf_event_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

uint64_t cycles_count_stop () {
    ioctl(perf_event_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    read(perf_event_fd, &perf_event_values, sizeof(perf_read_format_t));
    return perf_event_values.values[0].value;
}

void perf_done ()
{
    close(perf_event_fd);
    pfm_terminate ();
}