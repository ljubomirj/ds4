//
// Common functionality for C programs that can also go into C++ programs. Assume it will be compiled as a C, not C++.
//
// $Id: $

#ifndef PREFIX_FILE
#define PREFIX_FILE "Prefix.h"
#endif
#include PREFIX_FILE

#include <signal.h>
#include <stdarg.h>
#include <limits.h>
#include <unistd.h> // ftruncate, also needs _XOPEN_SOURCE>=500 
#include <sys/types.h>
#include <stdint.h> // intmax_t
#include <stddef.h> // ptrdiff_t

#include <math.h> // for NAN fpclassify
#include <float.h>
#include <fenv.h> // for feclearexcept

//For xcommon.h
#include <features.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h> // mremap MAP_HUGETLB, also needs _GNU_SOURCE in the prefix file
#include <syscall.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#include "xcommon.h"

// Should not be defined, should be unique to this file
#if defined(XFILE_VERB)
  #error "XFILE_VERB already defined, can't continue, fix the offender"
#endif
// Seems VERBL does not work => hard-code VERBX; Set VERB_LEVEL 1 then use VERB1 to output; VERB0 turns off
#define XFILE_VERB(x) VERB0(("XFILE:%s:%d: ",__FILE__,__LINE__)); VERQ0(x)
#define XFILE_VEXP(x) VEXP0(x)

#define FILL_ALLOC 0xcc
#define FILL_FREE 0xc9
#define MAX_ALLOC_BYTES (250LL * 1024LL * 1024LL * 1024LL)

static const char *sXerrorFile;
static long sXerrorLine;

// Assumes x86 set - add as appropriate for non-x86 here.
// https://stackoverflow.com/questions/173618/is-there-a-portable-equivalent-to-debugbreak-debugbreak
// For gcc use echo|cpp -dM to get cpp list all system specific macros.
void XbreakIntoDebugger(void) {
#if defined(_MSC_VER) || defined(__BORLANDC__)
  __asm { int 3 };
#elif defined(__i386__) ||  defined(_X86_) || defined(__x86_64__)
  __asm volatile("int3");
#elif defined(__GNUC__)
  __asm__ volatile("int3");
#elif defined(SIGTRAP)
  raise(SIGTRAP);
#else
  raise(SIGABRT);
#endif
}

void XerrorSet(const char *iFile, long iLine) {
  #pragma omp critical (Xerror)
  {
    sXerrorFile = iFile;
    sXerrorLine = iLine;
  }
}

// Choose how exit on non-zero exit code. 
// At runtime can just exit, or start the debugger or break into the debugger, or detect if under debugger and  break.
enum XPM {
  XPM_EXIT,
  XPM_BREAK_IF_IN_DEBUG,
  XPM_BREAK,
  XPM_COUNT
};
int sXexitPogramMode = XPM_BREAK_IF_IN_DEBUG;

// Return true if running under debugger, false otherwise.
// https://stackoverflow.com/questions/3596781/how-to-detect-if-the-current-process-is-being-run-by-gdb
// Can't be in the .hpp and be used here. While here can't use the xsnprintf for buffers.
int IsUnderDebugger(void) {
  // Returns 0 if TracePid exists and non-zero, 1 if it exists but zero, 2 if it was not found
  int q = 0;
  char buf[333];
  #pragma omp critical (Xerror)
  {
    snprintf(buf, sizeof(buf), "/bin/cat /proc/%d/status |/usr/bin/awk 'BEGIN{r=2}/TracerPid/{if($2==0){r=1}else{r=0}exit}END{exit r}'", getpid());
    buf[sizeof(buf) - 1] = '\0';
    q = system(buf);
  }
  return (q == 0);
}

// Exit the program
void XexitProgram(int iExitCode) {
  #pragma omp critical (Xerror)
  {
    if (iExitCode && sXexitPogramMode != XPM_EXIT) {
      if (sXexitPogramMode == XPM_BREAK) XbreakIntoDebugger();
      if (sXexitPogramMode == XPM_BREAK_IF_IN_DEBUG && IsUnderDebugger()) XbreakIntoDebugger();
    }
  }
  exit(iExitCode);
}

void XerrorRaise(const char *iFmt, ... ) {
  #pragma omp critical (Xerror)
  {
    fflush(stdout); fflush(stderr);
    printf("\nERROR at %s:%ld: ", sXerrorFile, sXerrorLine);
    if (iFmt) {
      va_list list;
      va_start(list, iFmt);
      vprintf(iFmt, list);
      va_end(list);
    }
    printf("\n");
    fflush(stdout);
    XexitProgram(EXIT_FAILURE);
  }
}

// XMEM

// Both debug and release versions abort if mem alloc fails.
// In addition the debug version overwites the freed memory,
// and keeps tally on the max ammount of memory used and aborts if exceeded.

// Must be defined, default to 0 for no output
#ifndef XALLOC_TRACE
  #define XALLOC_TRACE 0
#endif

// The actual trace output - prints ammounts alloced / freed in every call
#define XALLOC_VERB(x) VERBL(XALLOC_TRACE, ("    XALLOC:%s:%d: ", __FILE__, __LINE__)); VERBL(XALLOC_TRACE, x)

#if defined(DEBUG)

// Allow for MAX_ALLOC MB
static long long sBytesMax = MAX_ALLOC_BYTES;
static long long sBytesAlloced = 0;
static long long sBytesOverhead = 0;

// Report during malloc/free, or not
//#define XALLOC_REPORT do { char buf[333]; XallocReport(buf, 333); printf("XallocReport: %s\n", buf); } while(0)
#define XALLOC_REPORT

// Save the size in the first sizeof(size_t) bytes 
void *Xmalloc(size_t n) {
  void *p;
  size_t n1 = n + sizeof(size_t);
  #pragma omp critical (Xmalloc)
  {
    long long bytesAllocedNew = n1 + sBytesAlloced;
    //XASSERT(bytesAllocedNew <= sBytesMax); // use this to get stack trace
    XCHECK(bytesAllocedNew <= sBytesMax, ("Max mem alloc reached. Have %lld bytes, want another %lld bytes "
      "to a total of %lld bytes (%.3f MB). Will exceed max %lld bytes. Xmalloc will fail now.", 
      sBytesAlloced, (long long)n1, bytesAllocedNew, (float)bytesAllocedNew/1024.0/1024.0, sBytesMax));

    p = malloc(n1);
    XCHECK(p, ("malloc of %zu bytes failed", n1));

    memset(p, FILL_ALLOC, n1);
    sBytesAlloced = bytesAllocedNew;
    sBytesOverhead += (n1 - n);
    *(size_t *)p = n1;
    XALLOC_VERB(("Xmalloc: alloced %zu bytes at %p (user %p)\n", n1, p, (void *)((size_t *)p + 1)));
  }
  XALLOC_REPORT;
  return (void *)((size_t *)p + 1);
}

void *Xrealloc(void *p, size_t n) {
  void *p1;
  #pragma omp critical (Xmalloc)
  {
    size_t siz1 = n + sizeof(size_t);
    size_t siz = ((size_t *)p)[-1];
    long long bytesAllocedNew = sBytesAlloced - siz + siz1;
    XCHECK(bytesAllocedNew <= sBytesMax, ("Reallocing new %zu bytes instead old %zu bytes. "
      "Existing %lld bytes. New total %lld bytes (%.3f MB) will exceed max %lld bytes. Xrealloc will fail now.", 
      siz1, siz, sBytesAlloced, bytesAllocedNew, (float)bytesAllocedNew/1024.0/1024.0, sBytesMax));

    p1 = realloc((size_t *)p - 1, siz1);
    XCHECK(p1, ("realloc of %zu bytes failed at %p", siz1, p));

    sBytesAlloced = bytesAllocedNew;
    sBytesOverhead += (siz1 - siz);
    *(size_t *)p1 = siz1;
    XALLOC_VERB(("Xrealloc: alloced %zu bytes, freed %zu bytes, old %p, new %p\n", siz1, siz, (void *)((size_t *)p - 1), p1));
  }
  XALLOC_REPORT;
  return (void *)((size_t *)p1 + 1);
}

void Xfree(void *p) {
  XALLOC_VERB(("Xfree: user %p\n", p));
  if (p) {
    #pragma omp critical (Xmalloc)
    {
      size_t siz = ((size_t *)p)[-1];
      XCHECK((float)siz <= sBytesAlloced, ("Freeing %zu bytes. But have only %lld alloced. Xfree will fail now.", siz, sBytesAlloced));
      memset((size_t *)p - 1, FILL_FREE, siz);
      free((size_t *)p - 1);
      sBytesAlloced -= siz;
      sBytesOverhead -= sizeof(size_t);
      XALLOC_VERB(("Xfree: freed %zu bytes at %p\n", siz, (void *)((size_t *)p - 1)));
    }
  }
  XALLOC_REPORT;
} 

long long XallocedBytes(void) {
  long long a;
  #pragma omp critical (Xmalloc)
  {
    a = sBytesAlloced;
  }
  return a;
}

// Get the current limit with XmallocLimit(NULL), set a new limit x with XmallocBytes(& x) and return it (x will get the old limit)
long long XallocLimit(long long *ioLimitBytes) {
  if (ioLimitBytes) {
    #pragma omp critical (Xmalloc)
    {
      long long z = sBytesMax;
      sBytesMax = *ioLimitBytes;
      *ioLimitBytes = z;
    }
  }
  return sBytesMax;
}

char *XallocReport(char *oDst, size_t iDstSz) {

  #pragma omp critical (Xmalloc)
  {
    union u_malloc {
      unsigned char u_buf[32];
      unsigned char u_uchar;
      char u_char;
      int u_int;
      float u_float;
      double u_double;
    };
    union u_malloc u_al, u_fr;
    memset(& u_al, FILL_ALLOC, sizeof(u_al));
    memset(& u_fr, FILL_FREE, sizeof(u_fr));

    snprintf(oDst, iDstSz, "Alloced %.3f MB (%lld bytes), of which overhead %.3f MB (%lld bytes). Default max alloc is %.3f MB. Memory fill patterns on alloced is [uchar=0x%X char=%c int=0x%X int=%d float=%f float=%e double=%g], on freed is [uchar=0x%X char=%c int=0x%X int=%d float=%f float=%e double=%g]", (float)sBytesAlloced/1024.0/1024.0, sBytesAlloced, (float)sBytesOverhead/1024.0/1024.0, sBytesOverhead, (float)MAX_ALLOC_BYTES/1024.0/1024.0, u_al.u_uchar, u_al.u_char, u_al.u_int, u_al.u_int, u_al.u_float, u_al.u_float, u_al.u_double, u_fr.u_uchar, u_fr.u_char, u_fr.u_int, u_fr.u_int, u_fr.u_float, u_fr.u_float, u_fr.u_double);

    oDst[iDstSz - 1] = '\0';
  }
  return oDst;
}

#else /* #if defined(DEBUG) */

void *Xmalloc(size_t n) {
  void *p;
  #pragma omp critical (Xmalloc)
  {
    p = malloc(n);
    XCHECK(p, ("malloc of %zu bytes failed", n));
    XALLOC_VERB(("Xmalloc : user %p\n", p));
  }
  return p;
}

void *Xrealloc(void *p, size_t n) {
  void *q;
  XALLOC_VERB(("Xrealloc: user %p\n", p));
  #pragma omp critical (Xmalloc)
  {
    q = realloc(p, n);
  }
  XCHECK(q, ("realloc of %zu bytes failed", n));
  return q;
}

void Xfree(void *p) {
  XALLOC_VERB(("Xfree: user %p\n", p));
  if (p) {
    #pragma omp critical (Xmalloc)
    {
      free(p);
    }
  }
}

long long XallocedBytes(void) {
  return -1LL;
}

long long XallocLimit(long long *ioLimitBytes) {
  return -1LL;
}

char *XallocReport(char *oDst, size_t iDstSz) {
  #pragma omp critical (Xmalloc)
  {
    union u_malloc {
      unsigned char u_buf[32];
      unsigned char u_uchar;
      char u_char;
      int u_int;
      float u_float;
      double u_double;
    };
    union u_malloc u_al, u_fr;
    memset(& u_al, FILL_ALLOC, sizeof(u_al));
    memset(& u_fr, FILL_FREE, sizeof(u_fr));

    snprintf(oDst, iDstSz, "Non-debug version does not keep allocations tally. Default max alloc is %.3f MB. Memory fill patterns on alloced is [uchar=0x%X char=%c int=0x%X int=%d float=%f float=%e double=%g], on freed is [uchar=0x%X char=%c int=0x%X int=%d float=%f float=%e double=%g]", (float)MAX_ALLOC_BYTES/1024.0/1024.0, u_al.u_uchar, u_al.u_char, u_al.u_int, u_al.u_int, u_al.u_float, u_al.u_float, u_al.u_double, u_fr.u_uchar, u_fr.u_char, u_fr.u_int, u_fr.u_int, u_fr.u_float, u_fr.u_float, u_fr.u_double);
    oDst[iDstSz - 1] = '\0';
  }
  return oDst;
}

#endif // #if defined(DEBUG)

// Verb structure, one global for the whole program
static struct Verb sVerb = { .mLevel = 3, .mPrefix = { '\0' } };

const struct Verb *VerbGetGlobal(void) {
  return & sVerb;
}

int VerbIs(int v) { return VerbIs2(& sVerb, v); }
int VerbIs2(const struct Verb *V, int v) { 
  int v1;
  #pragma omp critical (Xerror)
  {
    v1 = V->mLevel; 
  }
  return v <= v1;
}

int VerbLevel(void) { return VerbLevel2(& sVerb); }
int VerbLevel2(const struct Verb *V) {
  int v;
  #pragma omp critical (Xerror)
  {
    v = V->mLevel; 
  }
  return v;
}

int VerbSet(int v) { return VerbSet2(& sVerb, v); }
int VerbSet2(struct Verb *V, int v) {
  int v1;
  #pragma omp critical (Verb)
  {
    v1 = V->mLevel;
    V->mLevel = v;
  }
  return v1;
}

// Danger - exposes the internals to the user - make sure the user is thread-safe
char *VerbPrefix(void) { return VerbPrefix2(& sVerb); }
char *VerbPrefix2(struct Verb *V) {
  return & V->mPrefix[0];
}

// Add prefix to the tail, e.g: push('ccc') on 'a.bb' results in 'a.bb.ccc'
int VerbPush(const char *iPrefix) { return VerbPush2(& sVerb, iPrefix); }
int VerbPush2(struct Verb *V, const char *iPrefix) {
  int rc = 0;
  #pragma omp critical (Verb)
  {
    int len = strlen(V->mPrefix);
    int len1 = strlen(iPrefix);
    if (len + len1 + 2 >= sizeof(V->mPrefix)) {
      rc = 2;
    } else {
      if (len > 0) V->mPrefix[len] = '.';
      for (int i = 0; i < len1; ++i) V->mPrefix[len + (len>0? 1: 0) + i] = (iPrefix[i] == '.'? '_': iPrefix[i]);
      V->mPrefix[len + (len>0? 1: 0) + len1] = '\0';
    }
  }
  return rc;
}

// Remove prefix from the tail, e.g: pop() on 'a.bb.ccc' results in 'a.bb'
int VerbPop(void) { return VerbPop2(& sVerb); }
int VerbPop2(struct Verb *V) {
  int len, rc = 0;
  #pragma omp critical (Verb)
  {
    len = strlen(V->mPrefix);
    if (len == 0) {
      rc = 2;
    } else {
      for (int i = len - 1; i >= 0; --i) {
        if (V->mPrefix[i] == '.') {
          V->mPrefix[i] = '\0';
          goto done;
        }
      }
      V->mPrefix[0] = '\0';
done:
      ;
    }
  }
  return rc;
}

// Print message if the verbosity level is high enough
int Verb(int v, const char *iFmt, ... ) {
  int rc = 0;
  #pragma omp critical (Verb)
  {
    if (VerbIs(v) && iFmt) {
      va_list list;
      fflush(stdout); fflush(stderr);
      if (*sVerb.mPrefix) printf("%s: ", sVerb.mPrefix);
      va_start(list, iFmt);
      rc = vprintf(iFmt, list);
      va_end(list);
      fflush(stdout);
    }
  }
  return rc;
}

int Verb2(const struct Verb *V, int v, const char *iFmt, ... ) {
  int rc = 0;
  #pragma omp critical (Verb)
  {
    if (VerbIs(v) && iFmt) {
      va_list list;
      fflush(stdout); fflush(stderr);
      if (*V->mPrefix) printf("%s: ", V->mPrefix);
      va_start(list, iFmt);
      rc = vprintf(iFmt, list);
      va_end(list);
      fflush(stdout);
    }
  }
  return rc;
}

// Print message - always, possibly with prefix
int Verbp(const char *iFmt, ... ) {
  int rc = 0;
  if (iFmt) {
    #pragma omp critical (Verb)
    {
      va_list list;
      fflush(stdout); fflush(stderr);
      if (*sVerb.mPrefix) printf("%s: ", sVerb.mPrefix);
      va_start(list, iFmt);
      rc = vprintf(iFmt, list);
      va_end(list);
      fflush(stdout);
    }
  }
  return rc;
}

int Verbp2(const struct Verb *V, const char *iFmt, ... ) {
  int rc = 0;
  #pragma omp critical (Verb)
  {
    if (iFmt) {
      va_list list;
      fflush(stdout); fflush(stderr);
      if (*V->mPrefix) printf("%s: ", V->mPrefix);
      va_start(list, iFmt);
      rc = vprintf(iFmt, list);
      va_end(list);
      fflush(stdout);
    }
  }
  return rc;
}

// Print message - always, and without any prefix
int Verbq(const char *iFmt, ... ) {
  int rc = 0;
  #pragma omp critical (Verb)
  {
    if (iFmt) {
      va_list list;
      fflush(stdout); fflush(stderr);
      va_start(list, iFmt);
      rc = vprintf(iFmt, list);
      va_end(list);
      fflush(stdout);
    }
  }
  return rc;
}

int Verbq2(const struct Verb *V, const char *iFmt, ... ) {
  int rc = 0;
  #pragma omp critical (Verb)
  {
    if (iFmt) {
      va_list list;
      fflush(stdout); fflush(stderr);
      va_start(list, iFmt);
      rc = vprintf(iFmt, list);
      va_end(list);
      fflush(stdout);
    }
  }
  return rc;
}

#if XASSERT_USE_XASSERT

XASSERT_DECLARE_IMAGE_NAME;

// FIXME: "xtern -e gdb" Does not work in execlp as the first arg has to be a file name only, without any args.
// The args "-e gdb" would net to be passed as args in the execlp args list, e.g:
// execlp("xterm", "xterm", "-e", "gdb", XASSERT_REFERENCE_IMAGE_NAME, mypid_as_string, 0);

// If fail to attach with ptrace: Operation not permitted try $ sudo sysctl -w kernel.yama.ptrace_scope=0.
// To make it permanent $ sudo vi /etc/sysctl.d/10-ptrace.conf so to have: kernel.yama.ptrace_scope = 0.

// In gdb add source dir for searching via (gdb) directory /path/to/dir. Keep adding multiple. Reset with lone (gdb) directory. Check ~/.gdbinit.
// Fix ddd font errors: $ sudo apt-get install xfonts-100dpi. Settings in ~/.ddd/init. Default 16pt fonts add Ddd*FontSize: 160.

// Choose at runtime whether to start the debugger or break into the debugger.
// If you can detect whether the pgm is running under the debugger, plug that inhere.
// Otherwise, once you are in the debugger, set sXassertDebugMode to 0 so that
// XASSERT breaks into the debugger instead of launching a new copy of the debugger.
// If -1 then 1st time called try to determine if under debugger or not automatically,
// and then set the global var to 0 or 1.
enum XDM {
  XDM_DETECT,
  XDM_DEBUG,
  XDM_BREAK,
  XDM_COUNT
};
const char *const sXDM_name[XDM_COUNT] = { "XDM_DETECT", "XDM_DEBUG", "XDM_BREAK" };
int sXassertDebugMode = XDM_DETECT;

//static char sDebuggerName[] = "xterm -e gdb";
static char sDebuggerName[] = "ddd";

void XassertDebug(const char *iBinary, const char *iSource, int iLine) {

  fflush(stdout); fflush(stderr);
  fprintf(stderr, "\n\n%s (pid %d): %s:%d: XASSERT_DEBUG triggered, debugger '%s' debug mode %s (%d). "
    "Once in (gdb) p IsUnderDebugger(), p  sXassertDebugMode=%d (%s).\n", 
    iBinary, getpid(), iSource, iLine, sDebuggerName, sXDM_name[sXassertDebugMode], sXassertDebugMode, XDM_BREAK, sXDM_name[XDM_BREAK]);
  fflush(stderr);

  pid_t mypid = getpid();
  char pid_s[33];
  snprintf(pid_s, sizeof(pid_s), "%d", mypid);
  pid_s[sizeof(pid_s) - 1] = '\0';

  pid_t childpid;
  if ((childpid = fork()) != 0) {
    // Once restarted, assume running under debugger, do not launch the debugger again next time XASSERT is trigerred
    if (sXassertDebugMode == XDM_DEBUG) sXassertDebugMode = XDM_BREAK;
    // The debugee - stop running
    kill(mypid, SIGSTOP);
  } else {
    // The debugger
    fprintf(stderr, "=================================================\n%s (pid %d): %s:%d: starting debugger %s to attach to pid %s. "
      "If I fail, try 'attach pid' manually once the debugger is up. Afterwards, in the debugger do:\n",
      iBinary, getpid(), __FILE__, __LINE__, sDebuggerName, pid_s);
    fprintf(stderr, 
      "\t1) (Debugger) Set a breakpoint right AFTER the assert line printed above (gdb 'b'). Usually backtrace the call stack (gdp 'bt') to get there (gdb 'up'). May have to switch thread (gdb list to find 'inf thr', switch 'thr 5').\n"
      "\t2) (Shell) Let the program run in background (sh 'bg'), so to run after unblocking (in the next step). Ensure (sh 'jobs') that its status is 'Running' not 'Stopped'.\n"
      "\t3) (Debugger) Do 'signal SIGCONT' (at (dbg) prompt) to unblock the debugged program. Then step next (gdb 'n') do debug.\n"
      "=================================================\n"); 
    fflush(stderr);

    // Maybe dettach some resources inherited form the parent process? TODO check Stevens Unix book.
    
    // Overlay the process. NB using 0 instead of NULL as last (terminating) arg causes warning: missing sentinel in function call
    execlp(sDebuggerName, sDebuggerName, iBinary, pid_s, NULL);

    // If here then execlp failed
    fprintf(stderr, "\n=================================================\n"
      "%s (pid %d): %s:%d: triggered from %s:%d, execlp failed, can not start the debugger, errno=%d (\"%s\"). "
      "If over X, maybe the DISPLAY envirnment is not set? I will abort the would-be-debuger process now - NB the debugee "
      "*should* be SIGSTOPped\n=================================================\n", 
      iBinary, getpid(), __FILE__, __LINE__, iSource, iLine, errno, strerror(errno));
    abort();
  }
}

void XassertStop(const char *iBinary, const char *iSource, int iLine) {
  pid_t mypid = getpid();
  fflush(stdout); fflush(stderr);
  fprintf(stderr, "\n\n%s (pid %d): %s:%d: XASSERT_STOP triggered\n", iBinary, mypid, iSource, iLine);
  fflush(stdout); fflush(stderr);
  // Stop running
  //kill(mypid, SIGSTOP); // if not under debugger
  XbreakIntoDebugger(); // if under debugger
}

void XassertCopy(const char *iSrc) {
  XASSERT_REFERENCE_IMAGE_NAME[0] = '\0';
  strncat(XASSERT_REFERENCE_IMAGE_NAME, iSrc, FILENAME_MAX - 1);
}

// Assert was triggerred - stop in debugger or start a debugger, attach to the process, and then stop
void XassertDebugOrStop(const char *iBinary, const char *iSource, int iLine) {
  if (sXassertDebugMode == XDM_DETECT) {
    sXassertDebugMode = (IsUnderDebugger()? XDM_BREAK: XDM_DEBUG);
  }
  switch (sXassertDebugMode) {
    case XDM_DEBUG: {
      XassertDebug(iBinary, iSource, iLine);
    } break;
    case XDM_BREAK: {
      XassertStop(iBinary, iSource, iLine);
    } break;
    default : {
      fflush(stderr);
      fprintf(stderr, "%s:%d: ERROR in error handling code, sXassertDebugMode=%d, abort()-ing now\n", __FILE__, __LINE__, sXassertDebugMode);
      fflush(stderr);
      abort();
    } break;
  }
}

#endif // #if USE_XASSERT

// Clone of strdup that uses XMALLOC for allocation
char *xmallocstrdup(const char *iSrc) {
  XASSERT(iSrc);
  size_t len = strlen(iSrc);
  char *dst = XMALLOC(char, 1 + len);
  strcpy(dst, iSrc);
  return dst;
}

// Create new empty structure, initialize with sensible values.
Xfile *XfileNew(const char *iPathname) {
  Xfile *xf = XMALLOC(Xfile, 1);
  // Copy over the name
  xf->mPathname = xmallocstrdup(iPathname);
  XfileInit(xf); // init all else to invalid values
  XFILE_VERB(("XfileNew %p on %s done\n", xf, xf->mPathname));
  return xf;
}

// Don't allow for un-initialized mXfile
Xmap *XmapNew(const char *iPathname) {
  Xmap *xm = XMALLOC(Xmap, 1);
  xm->mXfile = XfileNew(iPathname); // alloc + init
  XmapInit(xm); // initializes Xmap only, does not touch Xfile
  XFILE_VERB(("XmapNew %p on %s done\n", xm, xm->mXfile->mPathname));
  return xm;
}

// Takes ownership of Xfile
Xmap *XmapNewXfile(Xfile *xf) {
  Xmap *xm = XMALLOC(Xmap, 1);
  xm->mXfile = xf; // is already initialized
  XmapInit(xm); // initializes Xmap only, does not touch Xfile
  XFILE_VERB(("XmapNew %p on %s done\n", xm, xm->mXfile->mPathname));
  return xm;
}

// Initialize with sensible values
void XfileInit(Xfile *xf) {

  // fd
  xf->mFd = -1;
  xf->mFdFlags = 0;
  xf->mFdMode = 0;

  // FILE
  xf->mFile = NULL;
  xf->mFileMode = 0;

  // For mapping
  xf->mMapBlockSize = (size_t)sysconf(_SC_PAGESIZE);
  xf->mMapBlockBits = 0;
  xf->mMapBlockMask = 0;
  xf->mMapBlockMot = 0;

  // TODO do bit counting the smart rather than the pedestrian way. 
  // Atm shift until at or over, then check if math (at) pagesize.
  unsigned int a = 0, i = 0;
  for (a = 1, i = 0; i < CHAR_BIT * sizeof(xf->mMapBlockSize) && a < xf->mMapBlockSize; a <<= 1, ++i) {}
  if (a == xf->mMapBlockSize) {
    xf->mMapBlockBits = i;
    // All unsigned on both sides so bitops are ok?
    xf->mMapBlockMask = (1U << i) - 1;
    // NB mask is size_t which is unsigned, while off is off_t which is signed. 
    // Not sure if the unsigned rvalue will be destoryed when assigned to the signed lvalue?
    xf->mMapBlockMot = ((size_t)(-1)) & ~(xf->mMapBlockMask);
  }

  // Done on init only, don't demote to assert
  XCHECK(xf->mMapBlockSize > 0, ("Expect blksiz %zu > 0?", xf->mMapBlockSize));
  XCHECK(xf->mMapBlockBits == 0U || (1U << xf->mMapBlockBits) == xf->mMapBlockSize, ("Got 1 << blkbit %zu != blksiz %zu?", xf->mMapBlockBits, xf->mMapBlockSize));
  XCHECK(xf->mMapBlockMask == 0U || (xf->mMapBlockSize - 1 == xf->mMapBlockMask), ("Got blksiz-1 %zu != blkmsk %zu?", xf->mMapBlockSize - 1, xf->mMapBlockMask));
  XCHECK(xf->mMapBlockMask == 0U || ((xf->mMapBlockMask | xf->mMapBlockMot) == (size_t)(-1)), ("Got blk (msk %zu | mot %ld) != all ones?", xf->mMapBlockMask, xf->mMapBlockMot));
  XCHECK(xf->mMapBlockMot == 0 || (xf->mMapBlockMot % xf->mMapBlockSize == 0), ("Got blk mot %ld %% siz %zu = %d != 0?", xf->mMapBlockMot, xf->mMapBlockSize, xf->mMapBlockMot % xf->mMapBlockSize == 0));

  // Should be instantiated via some policy, but until then, rely on manual setting + runtime checking
#if XFILE_BLOCKSIZE_BITCAN
  XCHECK(xf->mMapBlockBits > 0, ("Must have blk siz %zu power of 2 if bitcan 1, but blk bit %zu", xf->mMapBlockSize, xf->mMapBlockBits)); 
#endif
  XFILE_VERB(("XfileInit %p blk siz %zu bit %zu msk %zu mot %jd done\n", xf, xf->mMapBlockSize, xf->mMapBlockBits, xf->mMapBlockMask, (intmax_t)(xf->mMapBlockMot)));
}

// NB MAP_FAILED is not 0, it is (void *)(-1). Does it mean 0 is ok? I doubt.
// Convention:
//  - 0 for ptr that was never mapped, i.e. is un-initialised
//  - MAP_FAILED for ptr mapped but expirienced some file error
// So both 0 and MAP_FAILED are invalid, just for a different reason.
// When checking the return arg of mapping ops check for MAP_FAILED.
// When testing whether ptr was initialized at all check against 0.
void XmapInit(Xmap *xm) {
  // Mmap
  xm->mMap = 0;
  xm->mMapLen = 0; // type size_t
  xm->mMapOff = 0; // type off_t or off64_t
  xm->mMapProt = 0;
  xm->mMapFlags = 0;
  XFILE_VERB(("MapInit %p done\n", xm));
}

// Unmap, fclose, close etc - do any cleanup needed for any live objects - then free the structure
void XfileDelete(Xfile *xf) {
  XCHECK(xf != 0, ("Null xf?")); // assert only? or accept 0?
  XfileShutdown(xf);
  XFREE(xf->mPathname); // leave it last so error reporting makes sense?
  XFREE(xf);
  XFILE_VERB(("XfileDelete %p done\n", xf));
}

// Delete Xmap
void XmapDelete(Xmap *xm) {
  XCHECK(xm != 0, ("Null xm?")); // assert only? accept 0?
  XmapShutdown(xm);         // Unmap everything ...
  XfileDelete(xm->mXfile);  // ... so the file can be closed ...
  XFREE(xm);                // ... then free
  XFILE_VERB(("XmapDelete %p done\n", xm));
}

// Delete Xmap but do not touch underlying file
void XmapDeleteXfile(Xmap *xm) {
  XCHECK(xm != 0, ("Null xm?")); // assert only? accept 0?
  XmapShutdown(xm); // Unmap everything ...
  XFREE(xm);        // ... then free
  XFILE_VERB(("XmapDeleteXfile %p done\n", xm));
}

void XfileShutdown(Xfile *xf) {
  XASSERT(xf);
  // FILE Cleanup
  if (xf->mFile) {
    int q = fclose(xf->mFile);
    XCHECK(q == 0, ("Fail %d fclose file %s mFile %p, error %d: %s", q, xf->mPathname, xf->mFile, errno, strerror(errno)));
    XFILE_VERB(("XfileShutdown %p fclose %p\n", xf, xf->mFile));
  }
  // fd Cleanup
  if (xf->mFd != -1) XfileFdClose(xf);
  XFILE_VERB(("XfileShutdown %p done\n", xf));
}

void XmapUnmap(Xmap *xm) {
  if (xm->mMap) {
    int q = munmap(xm->mMap, xm->mMapLen);
    XCHECK(q == 0, ("Fail %d un-mmap file %s map %jd(%lu)@%p error %d: %s", 
      q, xm->mXfile->mPathname, (intmax_t)(xm->mMapOff), xm->mMapLen, xm->mMap, errno, strerror(errno)));
    XFILE_VERB(("XmapUnmap %p unmap %jd(%lu)@%p\n", xm, (intmax_t)(xm->mMapOff), xm->mMapLen, xm->mMap));
    xm->mMap = 0;
  } else {
    XFILE_VERB(("Warning: XmapUnmap %p unmap %jd(%lu)@%p nothing to unmap\n", xm, (intmax_t)(xm->mMapOff), xm->mMapLen, xm->mMap));
  }
}

// Remap an existing offset at new size. Example:
// https://stackoverflow.com/questions/40398901/resize-posix-shared-memory-a-working-example
#if _GNU_SOURCE
// If supported, use a remap call
void XmapRemap(Xmap *xm, size_t iLen) {
  void *p = mremap(xm->mMap, xm->mMapLen, iLen, MREMAP_MAYMOVE);
  XCHECK(p, ("Failed re-mmap file %s map %jd(old %lu new %lu)@%p error %d: %s", xm->mXfile->mPathname, (intmax_t)(xm->mMapOff), xm->mMapLen, iLen, xm->mMap, errno, strerror(errno)));
  XFILE_VERB(("XmapRemap %p remap len %zu mremap %jd(old %zu new %zu)@(old %p new %p)\n", xm, iLen, (intmax_t)(xm->mMapOff), xm->mMapLen, iLen, xm->mMap, p));
  xm->mMap = p;
  xm->mMapLen = iLen;
}
#else
// If not supported, unmap and then map anew, but with the identical other params (offset, mode)
void XmapRemap(Xmap *xm, size_t iLen) {
  XFILE_VEXP(const void *p = xm->mMap); // will be zeroed in unmap, record to log
  XmapUnmap(xm);
  XFILE_VEXP(const size_t len = xm->mMapLen); // will be updated with the new len, record to log
  XmapFile(xm, xm->mMapOff, iLen, xm->mMapProt);
  XFILE_VERB(("XmapRemap %p remap len %zu un/map %jd(old %zu new %zu)@(old %p new %p)\n", xm, iLen, (intmax_t)(xm->mMapOff), len, xm->mMapLen, xm->mMap, p));
}
#endif

void XmapShutdown(Xmap *xm) {
  if (xm->mMap) XmapUnmap(xm); // Mmap cleanup
  XFILE_VERB(("XmapShutdown %p done\n", xm));
}

// Print content of Xfile struct, do error checking, return the total number of printed chars
int XfileFprint(Xfile *xf, FILE *f) {

  int qq = 0, q = 0;
  q = fprintf(f, "Xfile %p:\n", (void *)xf);
  XCHECK(q > 0, ("Fail %d fprintf1", q));
  qq += q;
  q = fprintf(f, "  Pathname = %s\n", xf->mPathname);
  XCHECK(q > 0, ("Fail %d fprintf2", q));
  qq += q;
  q = fprintf(f, "  Fd = %d\n", xf->mFd);
  XCHECK(q > 0, ("Fail %d fprintf3", q));
  qq += q;
  q = fprintf(f, "  File = %p\n", (void *)(xf->mFile));
  XCHECK(q > 0, ("Fail %d fprintf4", q));
  qq += q;
  return qq;
}

// Justified to stray away from the iArg convention, ie. xm and not iXm: xm is more like 
// "this", is a special argument (see it is always first arg as well) so exception is made.
int XmapFprint(Xmap *xm, FILE *f) {

  int qq = 0, q = 0;
  // Print underlying
  q = XfileFprint(xm->mXfile, f);
  q = fprintf(f, "  Mmap = %p\n", xm->mMap);
  XCHECK(q > 0, ("Fail %d fprintf5", q));
  qq += q;
  q = fprintf(f, "  MmapLen = %zu\n", xm->mMapLen);
  XCHECK(q > 0, ("Fail %d fprintf6", q));
  qq += q;
  q = fprintf(f, "  MmapOff = %jd\n", (intmax_t)(xm->mMapOff));
  XCHECK(q > 0, ("Fail %d fprintf7", q));
  qq += q;
  return qq;
}

int XfileFdOpen(Xfile *xf, int iFdFlags, int iFdMode) {
  XASSERT(xf->mFd == -1);
  int q = open(xf->mPathname, iFdFlags, iFdMode);
  XCHECK(q != -1, ("Fail open file %s, error %d: %s", xf->mPathname, errno, strerror(errno)));
  xf->mFd = q;
  xf->mFdFlags = iFdFlags;
  xf->mFdMode = iFdMode;
  XFILE_VERB(("XfileFdOpen %p file %s fd %d done\n", xf, xf->mPathname, xf->mFd));
  return xf->mFd;
}

void XfileFdClose(Xfile *xf) {
  XASSERT(xf->mFd != -1);
  int q = close(xf->mFd);
  XCHECK(q == 0, ("Fail %d close file %s fd %d, error %d: %s", q, xf->mPathname, xf->mFd, errno, strerror(errno)));
  XFILE_VERB(("XfileFdClose %p close file %s fd %d done\n", xf, xf->mPathname, xf->mFd));
  xf->mFd = -1;
}

size_t XfileFdWrite(Xfile *xf, void *p, size_t iSize) {
  XASSERT(p != 0);
  XASSERT(xf != 0);
  XASSERT(xf->mFd != -1);
  size_t q = write(xf->mFd, p, iSize);
  XCHECK(q == iSize, ("Fail %zu file %s fd %d write %zu, error %d: %s", q, xf->mPathname, xf->mFd, iSize, errno, strerror(errno)));
  return q;
}

off_t XfileFdSeekSet(Xfile *xf, off_t iOffset) {
  XASSERT(xf != 0);
  XASSERT(xf->mFd != -1);
  off_t q = lseek(xf->mFd, iOffset, SEEK_SET);
  XCHECK(q == iOffset, ("Fail %lu lseek file %s fd %d set %lu, error %d: %s", q, xf->mPathname, xf->mFd, iOffset, errno, strerror(errno)));
  return q;
}

off_t XfileFdSeekEnd(Xfile *xf, off_t iOffset) {
  XASSERT(xf != 0);
  XASSERT(xf->mFd != -1);
  off_t q = lseek(xf->mFd, iOffset, SEEK_END);
  XCHECK(q != (off_t)(-1), ("Fail %lu lseek file %s fd %d end %lu, error %d: %s", q, xf->mPathname, xf->mFd, iOffset, errno, strerror(errno)));
  return q;
}

#if __USE_LARGEFILE64
off64_t XfileFdSeekSet64(Xfile *xf, off64_t iOffset) {
  XASSERT(xf != 0);
  XASSERT(xf->mFd != -1);
  off64_t q = lseek64(xf->mFd, iOffset, SEEK_SET);
  XCHECK(q == iOffset, ("Fail %lu lseek64 file %s off %lu, error %d: %s", q, xf->mPathname, iOffset, errno, strerror(errno)));
  return q;
}

off64_t XfileFdSeekEnd64(Xfile *xf, off64_t iOffset) {
  XASSERT(xf != 0);
  XASSERT(xf->mFd != -1);
  off64_t q = lseek64(xf->mFd, iOffset, SEEK_END);
  XCHECK(q != (off64_t)(-1), ("Fail %jd lseek64 file %s fd %d end %jd, error %d: %s", (intmax_t)q, xf->mPathname, xf->mFd, (intmax_t)iOffset, errno, strerror(errno)));
  return q;
}
#endif

// Return 1 if file exists, 0 otherwise
int XfileExists(Xfile *xf) {
  struct stat stat_buf;
  int q = stat(xf->mPathname, & stat_buf);
  return (q == 0); 
}

// Return the current file length
size_t XfileLength(Xfile *xf) {
  struct stat stat_buf;
  int q = stat(xf->mPathname, & stat_buf);
  XFILE_VERB(("XfileLength stat rc %d siz %zu file %s fd %d (current errorno %d: %s)\n", q, stat_buf.st_size, xf->mPathname, xf->mFd, errno, strerror(errno)));
  XCHECK(q != -1, ("Fail %d stat file %s fd %d, error %d: %s", q, xf->mPathname, xf->mFd, errno, strerror(errno)));
  return stat_buf.st_size;
}

// Base mmap file from offset, with length, and protection mode. Record the arguments of the call.
void *XmapFile(Xmap *xm, off_t iOff, size_t iLen, int iProt) {
  XASSERT(xm);
  Xfile *const xf = xm->mXfile;
  XASSERT(xf);
  XASSERT(xf->mFd != -1);
  // Demote to ASSERT at some point?
  XCHECK(XFILE_REM4OFF(xf, iOff) == 0, ("Got off %jd not block %zu alligned? ", (intmax_t)iOff, xf->mMapBlockSize));

  // Guard against mmap-ing zero length segment, mmap fails. Put 0 for the pointer in that case. Do lots of error checking.
  void *p = 0;
  if (iLen > 0) {
     p = mmap(0, iLen, iProt, MAP_SHARED, xf->mFd, iOff);
    XCHECK(p, ("Zero ptr from mmap is assumed illegal?" "File %s fd %d off %jd len %zu prot %d, error %d: %s", xf->mPathname, xf->mFd, (intmax_t)iOff, iLen, iProt, errno, strerror(errno)));
    XCHECK(p != MAP_FAILED, ("Fail %p MAP_FAILED %p mmap file %s fd %d off %jd len %zu prot %d, error %d: %s", p, MAP_FAILED, xf->mPathname, xf->mFd, (intmax_t)iOff, iLen, iProt, errno, strerror(errno)));
  } else {
    XFILE_VERB(("Warninig: XmapFile %p off %jd len %zu prot %d map %p zero for zero length\n", xm, (intmax_t)iOff, iLen, iProt, p));
  }

  xm->mMap = p;
  xm->mMapLen = iLen;
  xm->mMapOff = iOff; // lvalue is off_t or off64_t
  xm->mMapProt = iProt;
  xm->mMapFlags = MAP_SHARED;
  XFILE_VERB(("XmapFile %p off %jd len %zu prot %d map %p\n", xm, (intmax_t)iOff, iLen, iProt, p));

  return p;
}

#if __USE_LARGEFILE64
// Offset is in units of blocksize => jumps the 32bit barrier, length of the window is still limited to 32bits.
void *XmapFile64(Xmap *xm, off64_t iOff, size_t iLen, int iProt) {
  XASSERT(xm);
  Xfile *const xf = xm->mXfile;
  XASSERT(xf);
  XASSERT(xf->mFd != -1);
  // Demote to ASSERT at some point
  XCHECK(XFILE_REM4OFF(xf, iOff) == 0, ("Got off64 %jd not block %zu alligned? ", (intmax_t)iOff, xf->mMapBlockSize));

  // Actually mmap2 is blocks, while mmap64 is off64_t, by looking into the signature. Do lots of error checking.
  void *p = mmap64(0, iLen, iProt, MAP_SHARED, xf->mFd, iOff);
  XCHECK(p, ("Zero ptr from mmap assumed illegal. File %s fd %d off %jd len %zu prot %d, error %d: %s", xf->mPathname, xf->mFd, (intmax_t)iOff, iLen, iProt, errno, strerror(errno)));
  XCHECK(p != MAP_FAILED, ("Fail mmap64 file %s off %jd len %zu prot %d, error %d: %s", xf->mPathname, (intmax_t)iOff, iLen, iProt, errno, strerror(errno)));

  xm->mMap = p;
  xm->mMapLen = iLen;
  // Both sides are off64_t
  xm->mMapOff = iOff;
  xm->mMapProt = iProt;
  xm->mMapFlags = MAP_SHARED;
  XFILE_VERB(("XmapFile64 %p off64 %jd len %zu prot %d map %p\n", xm, (intmax_t)iOff, iLen, iProt, p)); 
  return p;
}
#endif

// Open file, ensure large enough to fit the data window, write zero data in if needed, then mmap
void *XmapOpenRO(Xmap *xm, off_t iOff, size_t iLen) { return XmapOpen(xm, iOff, iLen, XfileReadOnly); }
void *XmapOpenRW(Xmap *xm, off_t iOff, size_t iLen) { return XmapOpen(xm, iOff, iLen, XfileReadWrite); }
void XfileModeSpecs(XfileMode iMode, int *oFlags, int *oMode, int *oProt) {
  switch (iMode) {
    case XfileReadWrite:
        *oFlags = O_RDWR | O_CREAT;
        *oMode = S_IRUSR | S_IWUSR;
        *oProt = PROT_READ | PROT_WRITE;
        break;
    case XfileReadOnly:
        *oFlags = O_RDONLY;
        *oMode = 0;
        *oProt = PROT_READ;
        break;
    default:
        XERROR(("Unknown mode %d", iMode));
  }
}

// Mmap a file segment
void *XmapOpen(Xmap *xm, off_t iOff, size_t iLen, XfileMode iMode) {
  XASSERT(xm);
  Xfile *const xf = xm->mXfile;
  XASSERT(xf);
  XASSERT(xf->mFd == -1);
  XASSERT(xf->mFile == NULL);
  int flags = 0, mode = 0, prot = 0;
  XfileModeSpecs(iMode, & flags, & mode, & prot);
  XfileFdOpen(xf, flags, mode);                       // open the file and ...
  void *const p = XmapFile(xm, iOff, iLen, prot); // ... map; the underlying fun checks for errors
  return p;
}

// Mmap the entire file
void *XmapOpenAllRO(Xmap *xm) { return XmapOpenAll(xm, XfileReadOnly); }
void *XmapOpenAllRW(Xmap *xm) { return XmapOpenAll(xm, XfileReadWrite); }
void *XmapOpenAll(Xmap *xm, XfileMode iMode) {
  XASSERT(xm);
  Xfile *const xf = xm->mXfile;
  XASSERT(xf);
  XASSERT(xf->mFd == -1);
  XASSERT(xf->mFile == NULL);
  int flags = 0, mode = 0, prot = 0;
  XfileModeSpecs(iMode, & flags, & mode, & prot);
  XfileFdOpen(xf, flags, mode);                             // open the file ...
  void *const p = XmapFile(xm, 0, XfileLength(xf), prot); // ... map; the underlying fun checks for errors
  return p;
}

#if __USE_LARGEFILE64
// Offset is in units of blocks, to jump over the 32bit barrier
void *XmapOpenRead64(Xmap *xm, off64_t iOff, size_t iLen) { return XmapOpen64(xm, iOff, iLen, XfileReadOnly); }
void *XmapOpenWrite64(Xmap *xm, off64_t iOff, size_t iLen) { return XmapOpen64(xm, iOff, iLen, XfileReadWrite); }
void *XmapOpen64(Xmap *xm, off64_t iOff, size_t iLen, XfileMode iMode) {
  XASSERT(xm);
  Xfile *const xf = xm->mXfile;
  XASSERT(xf);
  XASSERT(xf->mFd == -1);
  XASSERT(xf->mFile == NULL);
  int flags = 0, mode = 0, prot = 0;
  XfileModeSpecs(iMode, & flags, & mode, & prot);
  // Open file
  XfileFdOpen(xf, flags, mode);
  // Mmap - underlying function does the error checking
  return XmapFile64(xm, iOff, iLen, prot);
}
#endif

// Grow file by wlen bytes
#if 0
// Grow file by lseek past end, documented DOES NOT WORK
static void XfileGrowBy(Xfile *xf, off_t iWlen) {
  // Go at iWlen from end
  XfileFdSeekEnd(xf, iWlen);
}
#elif 0
// Grow file by writing empty buffer - silly but works 100%
static void XfileGrowBy(Xfile *xf, off_t iWlen) {
  // Go at end
  XfileFdSeekEnd(xf, 0);
  // Write zeros out, just seeking past end will not grow the file (documented). 
  // Write full sized blocks first (multiple), the remained next (one write).
  void *pb = XMALLOC(char, xf->mMapBlockSize);
  XMEMZERO(pb, xf->mMapBlockSize);
  // Write full sized blocks first
  for (int i = 0; i < iWlen/xf->mMapBlockSize; i++) XfileFdWrite(xf, pb, xf->mMapBlockSize);
  // Write reminder (less then a block) here if any
  size_t rem = iWlen % xf->mMapBlockSize;
  if (rem > 0) XfileFdWrite(xf, pb, rem);
  XFREE(pb);
}
#elif 0
// Grow file by lseek to new_end-1, then write 1 byte. Works, is fast,
// but possibly undefined - what content does the hole get?
static void XfileGrowBy(Xfile *xf, size_t iWlen) {
  // Don't fail for iWlen=0
  if (iWlen >= 1) {
    static const char buf[1] = { '\0' };
    XfileFdSeekEnd(xf, (off_t)(iWlen - 1)); // go at iWlen-1 from end ...
    XfileFdWrite(xf, (void *)buf, 1);       // ... then write a single byte
  }
}
#else
// Grow file by using ftruncate - it is documented, can truncate, fills any extra with \0
void XfileGrowBy(Xfile *xf, size_t iWlen) {
  XASSERT(xf->mFd != -1);
  // Get current length
  struct stat stat_buf;
  fstat(xf->mFd, & stat_buf);
  size_t new_size = iWlen + stat_buf.st_size;
  XCHECK(new_size >= stat_buf.st_size, ("File %s size %zd extend by %zu bytes overflew?", xf->mPathname, stat_buf.st_size, new_size));
  int q = ftruncate(xf->mFd, new_size);
  XCHECK(q == 0, ("Fail %d to truncate file %s to %zu bytes, error %d: %s", q, xf->mPathname, new_size, errno, strerror(errno)));
}
#endif

// Ensure file is of size iSize1+iSize2 in parts, without adding the two (for fear of overflow)
void XfileGrowToParts(Xfile *xf, size_t iSize1, size_t iSize2) {
  XASSERT(xf->mFd != -1);

  // Ensure there are iSize1+iSize2 bytes in the file
  struct stat stbuf;
  int q = fstat(xf->mFd, & stbuf);
  XCHECK(q == 0, ("Fail %d fstat file %s fd %d, error %d: %s", q, xf->mPathname, xf->mFd, errno, strerror(errno)));

  // Access size at our disposal
  size_t asiz = stbuf.st_size;

  // Can't sum iSize1+iSize2, may come to off64_t, can not use that here, so do it in parts
  if (asiz < iSize1) {

    // Grow file by the difference
    XfileGrowBy(xf, iSize1 - asiz);
    // Ensure there are iSize1+iSize2 bytes in the file
    q = fstat(xf->mFd, & stbuf);
    XCHECK(q == 0, ("Fail %d to fstat file %s, error %d: %s", q, xf->mPathname, errno, strerror(errno)));

    // Access size at disposal
    asiz = stbuf.st_size;
    // Should have enough for offset here
    XCHECK(asiz >= iSize1, ("Expect asiz %zu >= iSize1 %zu?", asiz, iSize1));
  }

  asiz -= iSize1;
  
  // Repeat the same as with size1, only for size2
  if (asiz < iSize2) {

    // Grow file by the difference
    XfileGrowBy(xf, iSize2 - asiz);
    // Ensure there are iSize2+iSize2 bytes in the file
    q = fstat(xf->mFd, & stbuf);
    XCHECK(q == 0, ("Fail %d to fstat file %s, error %d: %s", q, xf->mPathname, errno, strerror(errno)));
    // Access size at disposal
    asiz = stbuf.st_size;
    XCHECK(asiz >= iSize2, ("Expect asiz %zu >= iSize2 %zu?", asiz, iSize2)); // should have enough for offset here
  }
  XFILE_VERB(("Xfile %p XfileGrowToParts(iSize1 %zu iSize2 %zu)\n", xf, iSize1, iSize2));
}

#if __USE_LARGEFILE64
// Grow file by wlen bytes
#if 0
// Grow file by writing empty buffer - silly but works 100%
static void XfileGrow64(Xfile *xf, off64_t iWlen) {
  off64_t n = 0, rem = 0;
  char *pb = 0;
  // Go at end
  XfileFdSeekEnd64(xf, 0);
  // Write zeros out, just seeking past end will not grow the file (documented).
  // Write full sized blocks first (multiple), the remained next (one write).
  pb = XMALLOC(char, xf->mMapBlockSize);
  XMEMZERO(pb, xf->mMapBlockSize);
  // Write full sized blocks first. Warning: unsigned counter may lead to forever loop.
  // But the division by blocksize will ensure the upper bound is never max range. 
  // And it is .lt. op that terminates the loop, not .le. so doubly ok?
  for (n = 0; n < iWlen/xf->mMapBlockSize; n++) XfileFdWrite(xf, pb, xf->mMapBlockSize);
  // Write reminder (less then a block) here if any
  rem = iWlen % xf->mMapBlockSize;
  if (rem > 0) {
    // JIC blocksize ever grows t0 32bit+
    size_t rems = (size_t)rem;
    XASSERT((off64_t)rems == rem);
    XfileFdWrite(xf, pb, rems);
  }
  XFREE(pb);
}
#else
// Grow file by lseek to new_end-1, then write 1 byte. Works but possibly undefined - what content does the hole get?
static void XfileGrow64(Xfile *xf, off64_t iWlen) {
  // Don't fail for iWlen=0
  if (iWlen >= 1) {
    static const char buf[1] = { '\0' };
    XfileFdSeekEnd64(xf, iWlen - 1);  // go at iWlen-1 from end, and ...
    XfileFdWrite(xf, (void *)buf, 1); // ... write a single byte
  }
}
#endif

// Ensure file is of size iSize1+iSize2 in parts, without adding the two (for fear of overflow).
// There is no fstat64, nor ftell64, seems only way to ensure enough of file exists is to lseek64? Not
// sure if the holes created this way are mmap-able? Man only mentiones zeros will be read back from the
// hole till something is written in? But also mentiones that "... allows the file offset to be set beyond 
// the end of the file (but this does not change the size of the file)".
// So not possible to grow a file this way. Turns out there is fstat64, read the includes headers.
static void XfileGrowToParts64(Xfile *xf, off64_t iSize1, size_t iSize2) {
  XASSERT(xf);
  XASSERT(xf->mFd != -1);

  // Ensure there are iSize1+iSize2 bytes in the file
  struct stat64 stbuf;
  int q = fstat64(xf->mFd, & stbuf);
  XCHECK(q == 0, ("Fail %d fstat64 file %s fd %d, error %d: %s", q, xf->mPathname, xf->mFd, errno, strerror(errno)));

  // Access size at our disposal
  off64_t asiz = stbuf.st_size;
  // Can't sum iSize1+iSize2, may overflow off64_t, can not use that here. So do it in parts.
  if (asiz < iSize1) {

    XfileGrow64(xf, iSize1 - asiz); // grow file by the difference
    q = fstat64(xf->mFd, & stbuf);  // ensure there are iSize1+iSize2 bytes in the file
    XCHECK(q == 0, ("Fail %d to fstat64 file %s, error %d: %s", q, xf->mPathname, errno, strerror(errno)));

    // Access size at disposal, should have enough for offset here.
    asiz = stbuf.st_size;
    XCHECK(asiz >= iSize1, ("Expect asiz %jd >= iSize1 %jd", (intmax_t)asiz, (intmax_t)iSize1));
  }

  asiz -= iSize1;
  
  // Repeat the same as with size1, only for size2
  if (asiz < iSize2) {

    XfileGrow64(xf, iSize2 - asiz); // grow file by the difference
    q = fstat64(xf->mFd, & stbuf);  // ensure there are iSize2+iSize2 bytes in the file
    XCHECK(q == 0, ("Fail %d to fstat64 file %s, error %d: %s", q, xf->mPathname, errno, strerror(errno)));

    // Access size at disposal. Should have enough for offset here.
    asiz = stbuf.st_size;
    XCHECK(asiz >= iSize2, ("Expect asiz %jd >= iSize2 %zu", (intmax_t)asiz, iSize2));
  }
  XFILE_VERB(("XfileGrowToParts64 %p siz1 %jd siz2 %zu\n", xf, (intmax_t)iSize1, iSize2));
}
#endif

// Create new file with enough space if needed.
// If length of segment is 0 then mmap fails. So add special case of 0 length segment return 0 pointer.
void *XmapCreate(Xmap *xm, off_t iOff, size_t iLen) {
  XASSERT(xm);
  Xfile *const xf = xm->mXfile;
  XASSERT(xf);
  int flags = 0, mode = 0, prot = 0; // modes etc
  XfileModeSpecs(XfileReadWrite, & flags, & mode, & prot);
  if (xf->mFd == -1) XfileFdOpen(xf, flags, mode);  // open, then ...
  XfileGrowToParts(xf, (size_t)iOff, iLen);         // ... grow in size
  return XmapFile(xm, iOff, iLen, prot);            // map only if len non zero otherwise mmap fails; underlying fun checks for errors
}

#if __USE_LARGEFILE64
// Create new file with enough space if needed. Offset is 64bit and block alligned, to jump over the 32bit barrier.
void *XmapCreate64(Xmap *xm, off64_t iOff, size_t iLen) {
  XASSERT(xm);
  Xfile *const xf = xm->mXfile;
  XASSERT(xf);
  int flags = 0, mode = 0, prot = 0; // modes etc
  XfileModeSpecs(XfileReadWrite, & flags, & mode, & prot);
  if (xf->mFd == -1) XfileFdOpen(xf, flags, mode);  // open, then ...
  XfileGrowToParts64(xf, iOff, iLen);            // ... grow in size
  return XmapFile64(xm, iOff, iLen, prot);       // map, underlying function checks for errors
} 
#endif

// Does iStr string start with iPrefix string?
int xstrleft(const char *iStr, const char *iPrefix) {
  return strncmp(iStr, iPrefix, strlen(iPrefix));
}

// Skip any iSkip characters and return a pointer to the 1st non-iSkip char.
// The pointer returned may point to the terminating zero '\0' if all of iStr is used.
const char *xstrskip(const char *iStr, const char *iSkip) {
  return iStr + strspn(iStr, iSkip);
}

// Return true if file exists
int xisfile(const char *iPathname) {
  struct stat st;
  return stat(iPathname, & st) == 0;
}

// Return the difference, in seconds, between the current time relative to prior time
double xtimesec(struct timespec iFromTime) {
  // Time elapsed in seconds since 1-Jan-1970
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, & ts);
  double ts_delta = (ts.tv_sec - iFromTime.tv_sec) + (ts.tv_nsec - iFromTime.tv_nsec) / 1000000000.0;
  return ts_delta;
}

// Setup the floating point system such that
//  (1) Nans do not cause exceptions (all are "quiet nans")
//  (2) Nans propagate in arithmetic operations
// The compiler flag -ffast-math ("Improve FP speed by violating ANSI & IEEE rules") turns on various violations of IEEE 754 
// that may break assumptions below (see https://gcc.gnu.org/wiki/FloatingPointMath). Current best practise seem to be to 
// assert the assumptions we rely on during runtime.

// "Programs calling feclearexcept function shall ensure that pragma FENV_ACCESS is enabled for the call".
// But seems not supported on gcc/g++ so guard to avoid the "undefined pragma" warning.
#if !__GNUC__
#pragma STDC FENV_ACCESS on
#endif

// Enable exceptions on NaN only on some ops. NB it nees #pragma STDC FENV_ACCESS on
void xfpsetup(void) {

  //int q = feenableexcept(FE_INVALID | FE_OVERFLOW);
  // Disable any exceptions (NaN or otherwise), leave it to the code to do the right thing
  int q = feclearexcept(FE_ALL_EXCEPT);
  XCHECK(q == 0, ("Failed to setup floating point exceptions, error code %d", q));

  double d;
  d = nan("nan"); q = fpclassify(d); XCHECK(q == FP_NAN, ("Nan floating point %g classified as %d but FP_NAN is %d (isnan %d)", d, q, FP_NAN, isnan(d)));
  d = 0.0 / 0.0;  q = fpclassify(d); XCHECK(q == FP_NAN, ("Nan floating point %g classified as %d but FP_NAN is %d (isnan %d)", d, q, FP_NAN, isnan(d)));
  d = NAN;        q = fpclassify(d); XCHECK(q == FP_NAN, ("Nan floating point %g classified as %d but FP_NAN is %d (isnan %d)", d, q, FP_NAN, isnan(d)));
  // add more here as needed

}

// The treatmant of nans we desire for gcc should be guaranteed by -fno-finite-math-only but somethis does not work:
// https://stackoverflow.com/questions/38278300/successfully-enabling-fno-finite-math-only-on-nan-removal-method
// Given we mostly need to detect nans, maybe consider using own version of isnan? Not sure, so not exported yet,
// and comment out to avoid warning:
#if 0
int xisnan(float value) {

  union IEEE754_Single {
    float f;
    struct {
    #if BIG_ENDIAN
      uint32_t sign     : 1;
      uint32_t exponent : 8;
      uint32_t mantissa : 23;
    #else
      uint32_t mantissa : 23;
      uint32_t exponent : 8;
      uint32_t sign     : 1;
    #endif
    } bits;
  } u = { value };

  // In the IEEE 754 representation, a float is NaN when the mantissa is non-zero, and the exponent is all ones
  // (2^8 - 1 == 255).
  return (u.bits.mantissa != 0) && (u.bits.exponent == 255);
}
#endif

