#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <signal.h>

#include "xcommon.h"

static const char *sXerrorFile;
static long sXerrorLine;

void XbreakIntoDebugger(void) {
#if defined(__arm64__) || defined(__aarch64__)
    __builtin_trap();
#elif defined(__x86_64__)
    __asm__ volatile("int3");
#elif defined(SIGTRAP)
    raise(SIGTRAP);
#else
    raise(SIGABRT);
#endif
}

void XerrorSet(const char *file, long line) {
    sXerrorFile = file;
    sXerrorLine = line;
}

void XerrorRaise(const char *fmt, ...) {
    fflush(stdout);
    fflush(stderr);
    fprintf(stderr, "\nERROR at %s:%ld: ", sXerrorFile, sXerrorLine);
    if (fmt) {
        va_list list;
        va_start(list, fmt);
        vfprintf(stderr, fmt, list);
        va_end(list);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
    XbreakIntoDebugger();
    exit(EXIT_FAILURE);
}
