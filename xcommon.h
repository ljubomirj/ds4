#ifndef XCOMMON_H
#define XCOMMON_H

// Required includes (must appear before this header in the .c file):
//   #include <stdio.h>     // fprintf, stderr
//   #include <stdlib.h>    // malloc, free, exit
//   #include <string.h>    // memset
//   #include <stdint.h>    // int8_t, uint32_t, etc.
//   #include <stdarg.h>    // va_list (for function declarations)
//   #include <math.h>      // isnan (if used in XASSERT)

#define XSTR_(x) #x
#define XSTR(x) XSTR_(x)

#define XJOIN2(X,Y) X##Y
#define XJOIN1(X,Y) XJOIN2(X,Y)
#define XJOIN(X,Y) XJOIN1(X,Y)

typedef int8_t    i8;
typedef int16_t   i16;
typedef int32_t   i32;
typedef int64_t   i64;
typedef uint8_t   u8;
typedef uint16_t  u16;
typedef uint32_t  u32;
typedef uint64_t  u64;
typedef float     f32;
typedef double    f64;

#define FILL_ALLOC 0xCC
#define FILL_FREE  0xC9
#define XMEMSET_DEAD 0xC9

#ifdef L26F_DEBUG
  #define XMEMSET_INIT(ptr, bytes) memset((ptr), XMEMSET_DEAD, (bytes))
#else
  #define XMEMSET_INIT(ptr, bytes) memset((ptr), 0x00, (bytes))
#endif

#define XMEMZERO(dst, len) memset(dst, 0, len)

#if defined(__cplusplus)
extern "C" {
#endif

void XerrorSet(const char *file, long line);
void XerrorRaise(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void XbreakIntoDebugger(void);

#if defined(__cplusplus)
}
#endif

#define XERROR(x) do { XerrorSet(__FILE__,__LINE__); XerrorRaise x; } while(0)
#define XCHECK(x,y) XCHECK_TRUE(x,y)
#define XCHECK_TRUE(x,y) do { if(!(x)){XERROR(y);} } while(0)
#define XCHECK_FALSE(x,y) do { if(x){XERROR(y);} } while(0)

#ifdef L26F_DEBUG
  #define XASSERT(e) do { \
      if (!(e)) { \
          fprintf(stderr, "\nXASSERT failed: %s\n  at %s:%d\n", #e, __FILE__, __LINE__); \
          XbreakIntoDebugger(); \
      } \
  } while(0)
  #define XASSERT_DEBUG(e) XASSERT(e)
#else
  #define XASSERT(e) ((void)0)
  #define XASSERT_DEBUG(e) ((void)0)
#endif

#define XASSERT_STOP(e) do { \
    if (!(e)) { \
        fprintf(stderr, "\nXASSERT_STOP: %s\n  at %s:%d\n", #e, __FILE__, __LINE__); \
        XbreakIntoDebugger(); \
    } \
} while(0)

#define XSTOP do { XbreakIntoDebugger(); } while(0)

#define XASSERT_STATIC(expr,msg) \
  _Pragma("GCC diagnostic push") \
  _Pragma("GCC diagnostic ignored \"-Wunused-local-typedefs\"") \
  typedef char XJOIN(XJOIN(XJOIN(STATIC_ASSERT_,__LINE__),_),msg)[(expr)? 1: -1]; \
  _Pragma("GCC diagnostic pop")

#define XMALLOC(t,n) (t *)malloc(sizeof(t[n]))
#define XFREE(p) do { free(p); (p) = NULL; } while(0)

#define XLOG(...) fprintf(stderr, __VA_ARGS__)

#define XLOG_EVERY(n, i, total, ...) do { \
    if ((i) % (n) == 0 || (i) == (total) - 1) { \
        fprintf(stderr, "[%d/%d] ", (int)(i), (int)(total)); \
        fprintf(stderr, __VA_ARGS__); \
    } \
} while(0)

#endif
