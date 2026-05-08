// l26f: inference driver
#include <stdio.h>
#include <unistd.h>

int ds4_log_is_tty(FILE *f) {
    (void)f;
    return isatty(fileno(stderr)) ? 1 : 0;
}
