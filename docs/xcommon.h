//
// Included both in C and C++ programs. The C source files inlude this header (xcommon.h).
// The C++ source files include templated header (xcommon.hpp) in addition to this header (xcommon.h).
// Avoid include files .{h,hpp} themselves including other include files .{h,hpp}. Contain the #include-s in the .{c,cpp} files only.
//
// Conventions, rules of thumb. Break them for a reason & within reason, otherwise stick to them. When weighting alternatives consider
// code reading the highest priority. Make it easy to reason at the point of use, save the reader the effort of jumping to the definition.
// (comptime = compile time, compile-time; runtime = run time, run-time)
//   1) If code needs Stackoverflow search and language layering to be understood, then it is no good: change it until it is blindingly
//      obvious what is supposed to happen. Boring, bog standard - good. Dynamic and exciting - bad.
//   2) READABILITY IS THE STEPPING STONE TO CORRECTNESS: 1st understand what needs to happen, 2nd then assert, check etc that it does indeed happen.
//   3) Use const & for costly-to-copy types input arguments. Use pointers when arguments are changed (*not* non-const references) for output and
//      input/output arguments. Readability 1st - have the reader of the calling code know the variable changes in the function.
//   4) Use explicit in constructors with non-trivial arguments.
//   5) Support common headers prefix file Prefix.h or target specific one, in the project headers and the Makefile.
//   6) Public functions and classes upper cammel case, private lower cammel or better classic C snake case - easier to distinguish for a reader.
//      Member variables prefixed with "m", input args with "i", output arg with "o", input-output args "io".
//      Break the rule for 1st argument of function that actually operates on an "object" analogous to "this" in C++, e.g it is ok to have:
//      XmapXyz(Xmap *xm), XfileXyz(Xfile *xf), XyzPrint(FILE *f).
//   7) Single line comments can end without fullstop, multilines punctuated.
//   8) Preproc use #if, not #ifdef if possible, makes explicit the case of #define BLAH 0. But use #ifdef MSC_VER.
//   9) Pointer declaration is "T *p" not "T* p" nor "T * p". The * binds with p stronger than with T. 
//      However for references use "T & v" instead of "T &v" or "T& v" so not to miss the reference and aid readability.
//  10) Keep the "business code" in cammel case as per recommendations. The non-"business code", the scaffolding orthogonal to it, e.g
//      verbose logging, assertions, checking of runtime errors, have it in upper case e.g CHECK XASSERT V::P("logging msg"), so
//      to be able to mentally ignore tune out when reading the "business logic". READABILITY ENABLES CORRECTNESS.
//  11) Variable names in cammel case, with type info not part of the name. Make exception for complicated types like specific matrix
//      sizes, e.g Matrix2d data2_R2_XY(numRows2, sizeX + sizeY); or index into some array of elements, e.g long sec_index = oSyms->GetIndex(p);
//      or int fitDate_index = SnapLeft2<long>(iFitDate, iUni.mNumDayCal, iUni.mDayCal).
//  12) A vector operation done through array iteration for(i=0...){sth[i];} is ok in 1 line - just enclose the body in {}, even if a single line.
//  13) Prefer prefix operators over postfix (i.e ++i over i++) for weak reasons: standardization, easier reading ("increment i" over "i incremented").
// For a "C" class Abc:
//  14) AbcNew does alloc + init.
//  15) AbcInit does init only.
// "C" class Def extending Abc can:
//  16) in DefNew do AbcNew (which does alloc and init of Abc) + own alloc + own init via DefInit.
//  17) in DefInit do init only, not touching Abc.
//  18) Simlarly, AbcDelete does shutdown + free, while AbcShutdown does shutdown only.
//      Then DefDelete can do AbcDelete + DefShutdown + free, while DefShutdown does Def shutdown only.
// For Vector-s and Matrix-es naming conventions:
//  19) Names are lower letters like all other variables. Admitedly it is tempting to capitalize, but don't.
//  20) Do have the dimensions in short hand single letter capital affixed to the name, in a way to give the
//      reader some idea (even if incomplete and not 100% clear) about the size of the matrix - e.g:
//      Matrix data_R_XY(numRows, sizeX + sizeY);
//  21) Prefer "using" to "typedef" for the classes of matrix templates, e.g:
//      template class Matrix<double>;      using Matrix2d = Matrix<double>;      using AutoMatrix2d = AutoPtr<Matrix2d>;
//  22) In the specific classes to templates affix the dimensionality and the type of the element at end of name, e.g:
//      template class Vector<long>;        using Vector1l = Vector<long>;        using AutoVector1l = AutoPtr<Vector1l>;
//      template class Vector<char *>;      using Vector1pc = Vector<char *>;     using AutoVector1pc = AutoPtr<Vector1pc>;
// For Matrix classes that have separate slicing objects like MatrixRef, esp the Pmap ones:
//  23) Affix _ref to the slicing object, like this PmapMatrixRef<T> myMx_ref = myMx.GetRef() don't mixup with the name in cammel case.
//  24) Result of slicing operation also affixed same way PmapVectorRef<T> myMx_ref_i = myMx_ref[i] so clear where it came from and
//      separate from the name. All further derivatives have _deriv affixed to the name, in order of derivation. So keeps name separate,
//      and readible where it came from.
// For mmap-ed classes:
//  25) Keep the (small) metadata separate from the (big) mmaped data. Data in mmap can be easily accessed from matlab, numpy, R.
//      Having meta data mixed in with the data would make that difficult. Metadata can be hard coded by the user, or in a separate file.
//      It is up to the user how metadata should be persisted. In runtime metadata can be assumed to be in memory.
//  26) Have metadata members declared 1st at the top of the class, so to be able to quickly see what needs to be de/serialized.
// For old style C arrays use notation:
//  27) Allocated space is xyzSize const mNameSize=NUM_ELEMENTS; define mName[mNameSize]. If static array known at comptime then ok to leave out 
//      xyzSize use XNUMEL(mName) instead of mNameSize. XNUMEL is pointer-safe passing pointer instead of array results in comptime error.
//  28) Number of elements present is xyzCount e.g define mNameCount=0, assign add element mName[mNameCount++]=.
//  29) Do not use "number of" Num notation as mNumName or mNameNum as it is ambiguous (does "number" refer to the "size" or the "count"). 
//      Use Size and Count as suffix to make it explicit which one it is. Keep Num for genuine counters that are not refering to allocated size.
//  30) Encode matrix dimensions as zzz_AxB e.g. mImbalance_NxBxS, mPosition_TxS, mClosePrc_NxS. Keep even the 1s in vectors, e.g. mWei_1xS, and esp if 
//      it comes via slicing mWeight_NxS, e.g. mWei_1xS = mWeight_NxS[i]. With runtime defined matrix size, use U for the unknown runtime dim e.g. for 
//      comptime number of predictors P have float Xfit_UxP for matrix [U x P] with U known at runtime, P known at comptime.
//  31) If vectors are _1xN or _Nx1, use suffix _m or _n to denote m-th or n-th element of something, e.g. scalar xi = X_1xN[i].
//  32) For dynamic arrays in function arguments, use function(size_t N, T arr[N]) to pass the length of the array at runtime. NB it was the case
//      for runtime that was not possible, could only document as T arr[/*N*/]) for the human reader. C matrix via Variably-Modified Types (VMT)
//      https://gustedt.wordpress.com/2011/01/09/dont-be-afraid-of-variably-modified-types/, https://blog.joren.ga/vla-usecases:
//      #include <stdio.h>
//      #include <malloc.h>
//      // NB 'n', 'm' and 'b' are runtime variables holding the dimensions, not comptime constants.
//      // However, passing void funarr(int m,int n,int b,int(*arr_MxNxB)[m][n][b]){ works in C but not C++ - doh!
//      void funarr(int m, int n, int b, int (*arr_MxNxB)/*[m][n][b]*/) {
//      		for (int i = 0; i < m; ++i) {
//      			for (int j = 0; j < n; ++j) {
//      				for (int k = 0; k < b; ++k) {	(*arr_MxNxB)[i][j][k] = i + j + k; }
//      			}
//      		}
//      		for (int i = 0; i < m; ++i) {
//      			for (int j = 0; j < n; ++j) {
//      				for (int k = 0; k < b; ++k) { printf("arr_MxNxK[%d][%d][%d]=%d expect %d\n", i, j, k, (*arr_MxNxB)[i][j][k], i + j + k); }
//      			}
//      		}
//      }
//      void main() {
//      	int m = 3, n = 4, b = 5;
//      	//int (*arr_MxNxB)[m][n][b] = (decltype(arr_MxNxB))malloc(sizeof(*arr_MxNxB)); // decltype requires C++
//      	int (*arr_MxNxB)[m][n][b] = malloc(sizeof(*arr_MxNxB)); // in C we don't need to cast
//      	if (arr_MxNxB) {
//          funarr(m, n, b, arr_MxNxB);
//      		free(arr_MxNxB);
//      	}
//      	printf("arr_MxNxB m=%d n=%d b=%d sizeof(arr_MxNxB)=%zu sizeof(arr_MxNxB[0][0][0])=%zu\n", m, n, b, sizeof(*arr_MxNxB), sizeof((*arr_MxNxB)[0][0][0]));
//      }
//  33) Virtual functions in SubClass class must have override (for the virtual in SuperClass), but must not have virtual themselves.
// Space in templates arguments:
//  34) In definition add space as usual: template <class T, int N> class Marray : public MarrayBase {
//  35) In use of templated type assume everything is part of the tupe so do not add space: size_t len1 = Marray<T,N-1>::Sizeof(ap);
// Format specs for library types:
//  36) For size_t is %zu, for ssize_t is %zd
//  37) For off_t, off64_t and other signed types may not exist, workaround (intmax_t) casts to widest int then use %jd. Alternative is PRI  (and SCA, for scanf) 
//      macros in <cinttypes> or <inttypes.h>.
// Fields of data are fieldset, often have common datetime line:
//  36) Keep the concrete field type known at comptime. If not possible, then provide enum/int-to-type comptime map to cast-down any super-class 
//      (that abstracts the concrete type).
//  37) When operating on a fieldset, use local functions + some repetition in preference to 1) macros 2) boilerplate copy & paste & modify.
//      Readability is concern #1-2-3, as is a stepping stone to correctness.
// Conditional compilation formatting:
//  38) When conditional compilation with continuation in a new line, one tab space siffices for all continuations of the same line. No need for every 
//      conditional to add its own space. E.g. -
//      YES -->   some function(a1 arg1,                    NO -->    some function(a1 arg1,
//                #if COND1                                           #if COND1
//                  a2 arg2, a3 arg3,                                   a2 arg2, a3 arg3,
//                #endif                                              #endif
//                  a4 arg4,                                              a4 arg4,
//                #if COND2                                           #if COND2
//                  a5 arg5,                                                a5 arg5,
//                #endif                                              #endif
//                  a6 arg6, a7 arg7) {                                       a6 arg6, a7 arg7) {
//                }                                                   }
//  39) Comptime parametrisations versus runtime, comptime implemented via class templates or preprocessor, class templates and classes.
//      Comptime implemented via templated classess results in templates produced classes that to the C++ class system look completely 
//      unrelated. This then makes it hard to process all classes produced by a template in an uniform way within a single code block.
//      So if there is only ever going to be one realisation of a comptime parameter in one program, stick to the preprocessor. 
//      Example of this is NUM_RESP and NUM_HOR - it's unlikely that we will have different values for them in the same binary.
//      On the other end - think whether separate value of the parameters will make a new class. Is Alpha with different number of 
//      predictors with NUM_PRED in a template completely different class, unrelated in any way to all other template realisations,
//      or not? If the answer is "no", then keep the parameter as runtime, not comptime - even when the value is known at comptime.
//      Where comptime->runtime replacement replaces T arr[SIZE] = { NAN } with std::vector<T> vec, use vec.resize(size, NAN) to
//      a) initialize the vec-tor easily b) dimension vec such that subsequent code vec[i] does not need changing, as vec is 
//      already pre-allocated and no need for push_back() or similar to assign values to the vec-tor that replaced the arr-ay.
//
// $Id: $

#ifndef XCOMMON_H
#define XCOMMON_H

/* The includes needed by xcommon.h are:

#include <features.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/mman.h> // mremap MAP_HUGETLB, also needs _GNU_SOURCE in the prefix file
#include <syscall.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <limits.h> // for INT_MIN, INT_MAX and the like

*/

// If not defined elsewhere -
// https://stackoverflow.com/questions/6954284/why-prefer-template-based-static-assert-over-typedef-based-static-assert
#define STATIC_ASSERT(x) typedef char __STATIC_ASSERT__[(x)?1:-1]

#define XSTR_(x) #x
#define XSTR(x) XSTR_(x)

// NB convenient "X macro" for macro expansion of lists with X() transform applied locally.
// Documentation https://digitalmars.com/articles/b51.html, https://en.wikipedia.org/wiki/X_macro.

// If ptr null return empty string, otherwise ptr
#define X2X0(x) ((x)? (x): "")

// Null not-a-value out-of-band values per type. Modern C++ can do better: #define INT64_NAN std::numeric_limits<int64_t>::min().
// TODO can one ditch cpp #define completely and have constexpr int_64_t INT64_NAN instead?
// TODO have class or struct parametrised on type (float, double, int) that implements both the value and helper functions like isnan?
#define DOUBLE_NAN ((double)NAN)
#define FLOAT_NAN ((float)NAN)
#define INT_NAN INT_MIN
#define LONG_NAN LONG_MIN
#define LLONG_NAN LLONG_MIN
#define CHAR_NAC '\0'
#define INDEX_NAN INT_MIN

// Shortcut for C in lieue of https://en.cppreference.com/w/cpp/types/numeric_limits/infinity
#define INT_INF INT_MAX
#define LONG_INF LONG_MAX
#define LLONG_INF LLONG_MAX
#define FLOAT_INF HUGE_VALF
#define DOUBLE_INF HUGE_VAL

// Assume MIN=nan MIN+1=min ... 0 ... MAX=max so -MAX==(MIN+1). E.g. int16_t has -32768=NAN -32767=MIN ... 0 ... 32767=MAX.
STATIC_ASSERT(INT_NAN < -INT_MAX);
STATIC_ASSERT(LONG_NAN < -LONG_MAX);
STATIC_ASSERT(LLONG_NAN < -LLONG_MAX);
STATIC_ASSERT(INDEX_NAN < 0);

// Handy shortahnds reduce mental effort parsing - TODO switch the codebase over, develop further.
// From https://nullprogram.com/blog/2023/10/08/ via https://news.ycombinator.com/item?id=37815674.
typedef int8_t    i8;
typedef int32_t   i32;
typedef int64_t   i64;
typedef uint8_t   u8;
typedef uint32_t  u32;
typedef uint64_t  u64;
typedef float     f32;
typedef double    f64;
typedef int32_t   b32;
typedef uintptr_t uptr;
typedef char      c8;
typedef ptrdiff_t ssize;
typedef size_t    usize;

// Xmalloc for constant sizes
#define XMALLOC(t,n) (t *)Xmalloc(sizeof(t[n]))
#define XREALLOC(p,t,n) (t *)Xrealloc(p,sizeof(t[n]))
#define XFREE(p) Xfree(p)

// Needed if library is created that will be used to link to. In gcc: if -fvisibility=hidden then will *not* export by default
// => explicitly export only functions with __attribute__ ((visibility ("default"))) (in https://gcc.gnu.org/wiki/Visibility).
// In contrast, the local not-exported symbols are __attribute__ ((visibility ("hidden"))).
// On MS-Windows the exporting module must __declspec(dllexport), while the importing one must __declspec(dllimport).
#define EXPORT __attribute__ ((visibility ("default")))

#if __cplusplus
extern "C" {
#endif

EXPORT void *Xmalloc(size_t n);
EXPORT void *Xrealloc(void *p, size_t n);
EXPORT void Xfree(void *p);
EXPORT long long XallocedBytes(void);
EXPORT long long XallocLimit(long long * ioLimitBytes);
EXPORT char *XallocReport(char * oDst, size_t iDstSz);

#if __cplusplus
}
#endif

#define XALLOCED_BYTES (XallocedBytes())
#define XCOPY(target, source, n, type) memcpy(target, source, (n)*sizeof(type))

// XERROR - runtime checking
#define XERROR(x); {XerrorSet(__FILE__,__LINE__); XerrorRaise x ;}
#define XCHECK(x,y) XCHECK_TRUE(x,y)
#define XCHECK_TRUE(x,y); if(!(x)){XERROR(y);}
#define XCHECK_FALSE(x,y); if(x){XERROR(y);}
#define XCHECK_FILE(x,y); if(!(x)){XERROR(y);}

// Runtime checked versions of popular calls; NB if fopen failed and f is NULL, can not use ferror(f) to get errno
#define XFILE_FOPEN_CHECK(f, name, mode, modeMsg) \
  FILE *f = fopen(name, mode); \
  XCHECK_FILE(f, ("Failed to open file %s mode %s for %s, error# %d: %s", name, mode, modeMsg, errno, strerror(errno)))
#define XFOPEN_CHECK(f, name, mode, modeMsg) \
  f = fopen(name, mode); \
  XCHECK_FILE(f, ("Failed to open file %s mode %s for %s, error# %d: %s", name, mode, modeMsg, errno, strerror(errno)))

#if __cplusplus
extern "C" {
#endif

EXPORT void XerrorSet(const char *iFile, long iLine);
EXPORT void XerrorRaise(const char *iFmt, ... ) __attribute__((format(printf, 1, 2)));

#if __cplusplus
}
#endif

// Verbose or not printf.
// Verb is a function so use shortcut of logical operators to avoid evaluating the arguments when verbosity is not high enough to warrant a function call.
// VERB is a comptime macro that uses pre-processor to remove the verbose code at comptime. The comptime call for printing is the same as the runtime one.
// Haven't found a cpp way of doing VERB(X,("blah")), so settled for more pedestrian VERBX(("blah")).
// Additionally there is C++ class/namespace wrapper arround the VerbX functions, called as V::P(...), V::PP(...) etc.

struct Verb {
  int mLevel;
  char mPrefix[333];
};

#if __cplusplus
extern "C" {
#endif

EXPORT const struct Verb *VerbGetGlobal(void);
EXPORT int VerbIs(int v);
EXPORT int VerbIs2(const struct Verb *V, int v);
EXPORT int VerbLevel(void);
EXPORT int VerbLevel2(const struct Verb *V);
EXPORT int VerbSet(int v);
EXPORT int VerbSet2(struct Verb *V, int v);
EXPORT char *VerbPrefix(void);
EXPORT char *VerbPrefix2(struct Verb *V);
EXPORT int VerbPush(const char *iPrefix);
EXPORT int VerbPush2(struct Verb *V, const char *iPrefix);
EXPORT int VerbPop(void);
EXPORT int VerbPop2(struct Verb *V);
EXPORT int Verb(int v, const char *iFmt, ... ) __attribute__((format(printf, 2, 3)));
EXPORT int Verb2(const struct Verb *V, int v, const char *iFmt, ... ) __attribute__((format(printf, 3, 4)));
EXPORT int Verbp(const char *iFmt, ... ) __attribute__((format(printf, 1, 2)));
EXPORT int Verbp2(const struct Verb * /*unused*/, const char *iFmt, ... ) __attribute__((format(printf, 2, 3)));
EXPORT int Verbq(const char *iFmt, ... );
EXPORT int Verbqw(const struct Verb * /*unused*/, const char *iFmt, ... );

#if __cplusplus
}
#endif

// Compile time level [0,9], 0 switches off.
// Alas - DOES NOT WORK. Is is flawed as cpp does not do another pass in
//  #define VERB(v, f) VERB ## v ## (f)
// So do it the pedestrian way, leave to the user to struggle with VERB0, VERB1, etc.
// Levels (add at will):
//  9 - function tracing, arguments print
//  8 - diags like memory, size etc
//  ................
//  0 - quet, no VERB comptime output

#ifndef VERB_LEVEL
  #define VERB_LEVEL 1
#endif

// Extend here for levels higher then 9 (unlikely?)
#if VERB_LEVEL < 0 || VERB_LEVEL > 9
  #error Unsupported VERB_LEVEL not in [0,9] - FIXME
#endif

#if VERB_LEVEL < 9
  #define VERB9(f) (void)0
  #define VERQ9(f) (void)0
  #define VEXP9(x) (void)0
#else
  #define VERB9(f) (Verbp f)
  #define VERQ9(f) (Verbq f)
  #define VEXP9(x) x
#endif
#if VERB_LEVEL < 8
  #define VERB8(f) (void)0
  #define VERQ8(f) (void)0
  #define VEXP8(x) (void)0
#else
  #define VERB8(f) (Verbp f)
  #define VERQ8(f) (Verbq f)
  #define VEXP8(x) x
#endif
#if VERB_LEVEL < 7
  #define VERB7(f) (void)0
  #define VERQ7(f) (void)0
  #define VEXP7(x) (void)0
#else
  #define VERB7(f) (Verbp f)
  #define VERB7(f) (Verbq f)
  #define VEXP7(x) x
#endif
#if VERB_LEVEL < 6
  #define VERB6(f) (void)0
  #define VERQ6(f) (void)0
  #define VEXP6(x) (void)0
#else
  #define VERB6(f) (Verbp f)
  #define VERQ6(f) (Verbq f)
  #define VEXP6(x) x
#endif
#if VERB_LEVEL < 5
  #define VERB5(f) (void)0
  #define VERQ5(f) (void)0
  #define VEXP5(x) (void)0
#else
  #define VERB5(f) (Verbp f)
  #define VERQ5(f) (Verbq f)
  #define VEXP5(x) x
#endif
#if VERB_LEVEL < 4
  #define VERB4(f) (void)0
  #define VERQ4(f) (void)0
  #define VEXP4(x) (void)0
#else
  #define VERB4(f) (Verbp f)
  #define VERQ4(f) (Verbq f)
  #define VEXP4(x) x
#endif
#if VERB_LEVEL < 3
  #define VERB3(f) (void)0
  #define VERQ3(f) (void)0
  #define VEXP3(x) (void)0
#else
  #define VERB3(f) (Verbp f)
  #define VERQ3(f) (Verbq f)
  #define VEXP3(x) x
#endif
#if VERB_LEVEL < 2
  #define VERB2(f) (void)0
  #define VERQ2(f) (void)0
  #define VEXP2(x) (void)0
#else
  #define VERB2(f) (Verbp f)
  #define VERQ2(f) (Verbq f)
  #define VEXP2(x) x
#endif
#if VERB_LEVEL < 1
  #define VERB1(f) (void)0
  #define VERQ1(f) (void)0
  #define VEXP1(x) (void)0
#else
  #define VERB1(f) (Verbp f)
  #define VERQ1(f) (Verbq f)
  #define VEXP1(x) x
#endif
// Level 0 always unconditionally resovles to nothing
#define VERB0(f) (void)0
#define VERQ0(f) (void)0
#define VEXP0(f) (void)0

// VERBL variant with 2 args, 1st one the level - alas does not work, use manual VERB0 - VERB9 i.e VERBL instead of VERB(L.
// NB seems to work ok in .c files, just not in .cpp files??
#define VERBLLL(x) VERBLLLL(x)
#define VERBLLLL(x) VERB##x
#define VERBL(x,y) VERBLLL(x) (y)

// Export symbols/not explicitly, after -fvisibility=hidden to hide them; from https://gcc.gnu.org/wiki/Visibility
// Generic helper definitions for shared library support.
#if defined _WIN32 || defined __CYGWIN__
  #define XSYMBOL_IMPORT __declspec(dllimport)
  #define XSYMBOL_EXPORT __declspec(dllexport)
  #define XSYMBOL_LOCAL
#else
  #if __GNUC__ >= 4
    #define XSYMBOL_IMPORT __attribute__ ((visibility ("default")))
    #define XSYMBOL_EXPORT __attribute__ ((visibility ("default")))
    #define XSYMBOL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define XSYMBOL_IMPORT
    #define XSYMBOL_EXPORT
    #define XSYMBOL_LOCAL
  #endif
#endif

// XASSERT - crash in release build or start debugger & stop in debug build
#if !defined(USE_XASSERT)
  //#warning Did not define USE_XASSERT
#endif
#if !defined(__unix__)
  //#warning Did not define __unix__
#endif
#if defined(USE_XASSERT) && defined(__unix__)
  #define XASSERT_USE_XASSERT 1
#else
  //#warning Did not define XASSERT_USE_XASSERT
#endif

// Both PATH_MAX (POSIX) and FILENAME_MAX (libc) maybe broken by default as impossible to determine => use hardcoded.
// Don't really need "max", just "larger than potentially largest" for buffers that can hold a filepath name.
#define XFILENAMEMAX 4096

#if XASSERT_USE_XASSERT

  #define XASSERT_DEFINE_IMAGE_NAME XSYMBOL_EXPORT char gXassertImageName[XFILENAMEMAX]
  #define XASSERT_DECLARE_IMAGE_NAME extern XSYMBOL_IMPORT char gXassertImageName[XFILENAMEMAX]
  // It will be defined in the executable
  XASSERT_DECLARE_IMAGE_NAME;
  #define XASSERT_COPY_IMAGE_NAME(a) XassertCopy(a)
  #define XASSERT_REFERENCE_IMAGE_NAME ((char *)gXassertImageName)

  #define XASSERT_DEBUG(e) ((e)? ((void)0): XassertDebug(XASSERT_REFERENCE_IMAGE_NAME,__FILE__,__LINE__))
  #define XASSERT_STOP(e) ((e)? ((void)0): XassertStop(XASSERT_REFERENCE_IMAGE_NAME,__FILE__,__LINE__))
  #if NDEBUG
    #define XASSERT(e) ((void)0)
  #elif XASSERT_ALWAYS_STOP
    #define XASSERT(e) ((e)? ((void)0): XassertStop(XASSERT_REFERENCE_IMAGE_NAME,__FILE__,__LINE__))
  #else
    #define XASSERT(e) ((e)? ((void)0): XassertDebugOrStop(XASSERT_REFERENCE_IMAGE_NAME,__FILE__,__LINE__))
  #endif
  #define XSTOP XassertStop(XASSERT_REFERENCE_IMAGE_NAME,__FILE__,__LINE__)
  #define XBREAK XbreakIntoDebugger()

  // Just insert a statement
  #define XASSERT_DO(s) (s)

  #if __cplusplus
  extern "C" {
  #endif

  EXPORT void XassertDebug(const char *binary, const char *source, int line);
  EXPORT void XassertStop(const char *binary, const char *source, int line);
  EXPORT void XassertCopy(const char *src);
  EXPORT void XassertDebugOrStop(const char *binary, const char *source, int line);
  EXPORT void XbreakIntoDebugger(void);

  #if __cplusplus
  }
  #endif

#else

  #include <assert.h>

  #define XASSERT(e) assert(e)
  #define XASSERT_DEBUG(e) assert(e)
  #define XASSERT_STOP(e) assert(e)
  #define XSTOP ((void)0)
  #define XBREAK ((void)0)
  #if DEBUG
    #define XASSERT_DO(s) (s)
  #else
    #define XASSERT_DO(s) ((void)0)
  #endif
  #define XASSERT_DEFINE_IMAGE_NAME
  #define XASSERT_DECLARE_IMAGE_NAME
  #define XASSERT_COPY_IMAGE_NAME(a) ((void)0)
  #define XASSERT_REFERENCE_IMAGE_NAME ((void)0)

  //#warning "Not defined XASSERT"

#endif

// Shorthand
#if DEBUG
  #define IFDEBUG(a,b) a
#else
  #define IFDEBUG(a,b) b
#endif

// XMEM
#define XMEMSET(dst, c, len) memset(dst, c, len)
#define XMEMZERO(dst, len) XMEMSET(dst, 0, len)

// From Boost - apparently two levels of indirection needed depending on whether called from another macro or not.
// First level of indirection needed always, for __LINE__ expansion.
#define XJOIN2(X,Y) X##Y
#define XJOIN1(X,Y) XJOIN2(X,Y)
#define XJOIN(X,Y) XJOIN1(X,Y)

// Compile time assert. Not ideal. Use 'gcc -E' to check preproc output, 'gcc -S' to check assembly.
#if __cplusplus
  // Fancy by Alexandrescu - uses template specialisation to match good/bad:
  #if 0
    template<bool> struct CompileTimeChecker {};
    template<> struct CompileTimeChecker<true> {};
    #define XASSERT_STATIC(expr,msg) (CompileTimeChecker<(expr) != 0>())
  #endif
    // Above does not make allowance for user message, so v2:
  #if 0
    template<bool> struct CompileTimeChecker { CompileTimeChecker(...); };
    template<> struct CompileTimeChecker<false> {};
    #define XASSERT_STATIC(expr, msg) \
    { \
        class ERROR_##msg {}; \
        (void)sizeof CompileTimeChecker<(expr) != 0>((ERROR_##msg())); \
    }
  #endif
  // But above does not allow for global scope, so fallback to the simpler C version:
  // ...
#else
  // Traditional by Van Horn relies on array[0] being illegal.
  // Check http://www.jaggersoft.com/pubs/CVu11_3.html for more options - can use any const-expr in the grammar for same effect.
  // But actually array[0] compiles in "default" C gcc => use -1 to def fail. Does not work at global scope, must be inside function.
  // Error message must be identifier, ex:
  //  XASSERT_STATIC(sizeof(T)>=sizeof(int), Type_T_must_be_at_least_int_wide);
  // Alternative is to generate type, but can't generate unique name => back to the original generate variable:
  //#define XASSERT_STATIC(expr,msg) { char XJOIN(XJOIN(XJOIN(ERROR_at_,__LINE__),_),msg)[(expr)? 1: -1]; }
  // Without scope so works in global scope:
  //#define XASSERT_STATIC(expr,msg) char XJOIN(XJOIN(XJOIN(ERROR_at_,__LINE__),_),msg)[(expr)? 1: -1];
  // But variable leaves [1] turds when assert is true, violating the requirement to evaluate to nothing when
  // true? So back to the typedef solution, emits nothing in the object file:
  // ...
#endif
// The error message must be an identifier, ex: XASSERT_STATIC(sizeof(T)>=sizeof(int), Type_T_must_be_at_least_int_wide);
// Suspend Warning temporarily locally: https://stackoverflow.com/questions/3378560/how-to-disable-gcc-warnings-for-a-few-lines-of-code
// Optionally add __INCLUDE_LEVEL__ to the unque typdef identifier in addition to __LINE__.
#define XASSERT_STATIC(expr,msg) \
  _Pragma("GCC diagnostic push") \
  _Pragma("GCC diagnostic ignored \"-Wunused-local-typedefs\"") \
  typedef char XJOIN(XJOIN(XJOIN(STATIC_ASSERT_,__LINE__),_),msg)[(expr)? 1: -1]; \
  _Pragma("GCC diagnostic pop")


// Memory pool that can be protected
enum MemoryAccessMode {
  NoAccess = 1,
  ReadOnly = 2,
  ReadWrite = 3
};

void MemoryAccessSet(const void *addr, size_t size, int mode);
void *MemoryAccessAllocate(size_t size);
void MemoryAccessFree(void *addr, size_t size);

// Placeholder for all things file related - open, fopen, mmap. Does error checking on all operations. Lightweight, barely wrappers.
// Ideally, want "C"
//      Xfile *XfileGet(const char *pathname);
// as well as C++
//     Xfile blah;
// as well as
//     AutoPtr<Xfile>
// and in C++ case with a wrapper class. Do the "C" part first, see how goes.
// Super light, just op + error checking + placeholder for all data related to the same file, nothing else. If in doubt, DONTDO.
//
// The convention on the pointer returned by mmap is:
//    - 0 for ptr that was never mapped, i.e. is un-initialised
//    - MAP_FAILED for ptr mapped but expirienced some file error
// So both 0 and MAP_FAILED are invalid, just for a different reason. When checking the return arg of mapping ops check
// for MAP_FAILED. When testing whther ptr was initialised at all check against 0.

// Seto to 1 if safe to assume pagefile blocksize will be power of 2.  Will be checked at runtime if true.
#ifndef XFILE_BLOCKSIZE_BITCAN
  #define XFILE_BLOCKSIZE_BITCAN 1
#endif

// Default to go with widest, but can redefine ex: for testing
#ifndef XMAP_OFFT
  #define XMAP_OFFSET_TYPE_64 __USE_LARGEFILE64
#endif

// Offsets off_t are signed, sizes size_t are unsigned, signed+unsigned=unsigned b/c signed is promoted to unsigned
#if XMAP_OFFSET_TYPE_64
  typedef off64_t XoffT;
#else
  typedef off_t XoffT;
#endif

// File info of name, open, fopen, mmap etc designation
typedef struct Xfile {
  // For all
  char *mPathname;
  // open Related
  int mFd;
  int mFdFlags;
  mode_t mFdMode;
  // fopen Related
  FILE *mFile;
  int mFileMode;
  // Same for all maps
  size_t mMapBlockSize;
  size_t mMapBlockBits;
  size_t mMapBlockMask;
  off_t mMapBlockMot;
} Xfile;

// Hold info about one memory map + underlying file.
// Given multiple memory maps over same file are always in danger of clashing (are they?), makes sense to
// wrap them inside a memory map or a pool or some other collection of maps? So non mapped access is possible
// via Xfile (no Xmap there), but map of any kind must go through the wrapper which synchronises access as
// it sees fit? Handy for monitoring mmap io $ sudo iotop -aoP -d 5.
typedef struct Xmap {
  Xfile *mXfile;
  void *mMap;
  size_t mMapLen;
  XoffT mMapOff;
  int mMapProt;
  int mMapFlags;
} Xmap;

// Both Xfile and Xmap are really classes, only not made explicit because want them to stay C/not stray C++.
// This is further reflected in all functions having Xfile/Xmap prefix, Xmap functions calling the Xfile ones
// on the contained mXfile, and 1st arguemnt being special this-ptr and breaking the iArg convention (is xm, xp etc).

// TODO FIXME mMapBlockSize will be power of 2, use this  to convert the macros below into bit ops rather then
// arithmetic ones. NB off will be off_t or off64_t, don't expect size_t.
//
#if XFILE_BLOCKSIZE_BITCAN
  #define XFILE_BLK4OFF(xf,off) ((off) >> (xf->mMapBlockBits))
  #define XFILE_REM4OFF(xf,off) ((off) & (xf->mMapBlockMask))
  // Get offset from desired, unalligned size => maps to the *next* slot
  #define XFILE_SIZ2OFF(xf,siz) ((off_t)((XFILE_BLK4OFF(xf,siz) + (XFILE_REM4OFF(xf,siz)? 1: 0)) << (xf->mMapBlockBits)))
  // Get offset from desired offset => maps to the *previois* slot
  #define XFILE_OFF4OFF(xf,off) ((off_t)((off) & xf->mMapBlockMot))
  // Get offset for n items, each size siz => maps to the *next* slot
  #define XFILE_SIZN2OFF(xf,siz,n) (XFILE_SIZ2OFF(xf,siz)*(n))
  // Get 64bit offset from desired, unalligned size => maps to the *next* slot
  #define XFILE_SIZ2OFF64(xf,siz) ((off64_t)((XFILE_BLK4OFF(xf,siz) + (XFILE_REM4OFF(xf,siz)? 1: 0)) << (xf->mMapBlockBits)))
  // Get 64bit offset from desired offset => maps to the *previois* slot
  #define XFILE_OFF4OFF64(xf,off) ((off64_t)((off) & xf->mMapBlockMot))
  // Get 64bit offset for n items, each size siz => maps to the *next* slot
  #define XFILE_SIZN2OFF64(xf,siz,n) (XFILE_SIZ2OFF64(xf,siz)*(n))
#else
  #define XFILE_BLK4OFF(xf,off) ((off) / (size_t)(xf->mMapBlockSize))
  #define XFILE_REM4OFF(xf,off) ((off) % (size_t)(xf->mMapBlockSize))
  // Get offset from desired, unalligned size => maps to the *next* slot
  #define XFILE_SIZ2OFF(xf,siz) ((off_t)((XFILE_BLK4OFF(xf,siz) + (XFILE_REM4OFF(xf,siz)? 1: 0)) *(off_t)(xf->mMapBlockSize)))
  // Get offset from desired offset => maps to the *previois* slot
  #define XFILE_OFF4OFF(xf,off) ((off_t)((XFILE_BLK4OFF(xf,off)) *(off_t)(xf->mMapBlockSize)))
  // Get offset for n items, each size siz => maps to the *next* slot
  #define XFILE_SIZN2OFF(xf,siz,n) (XFILE_SIZ2OFF(xf,siz)*(n))
  // Get 64bit offset from desired, unalligned size => maps to the *next* slot
  #define XFILE_SIZ2OFF64(xf,siz) ((off64_t)((XFILE_BLK4OFF(xf,siz) + (XFILE_REM4OFF(xf,siz)? 1: 0)) *(off64_t)(xf->mMapBlockSize)))
  // Get 64bit offset from desired offset => maps to the *previois* slot
  #define XFILE_OFF4OFF64(xf,off) ((off64_t)((XFILE_BLK4OFF(xf,off)) *(off64_t)(xf->mMapBlockSize)))
  // Get 64bit offset for n items, each size siz => maps to the *next* slot
  #define XFILE_SIZN2OFF64(xf,siz,n) (XFILE_SIZ2OFF64(xf,siz)*(n))
#endif

typedef enum {
  XfileReadWrite = 1,
  XfileReadOnly = 2
} XfileMode;

#if __cplusplus
extern "C" {
#endif

// Clone of strdup that uses XMALLOC for allocation
EXPORT char * xmallocstrdup(const char * iSrc);

// Xfile
EXPORT Xfile *XfileNew(const char *iPathname);
EXPORT void XfileInit(Xfile *xf);
EXPORT void XfileDelete(Xfile *xf);
EXPORT void XfileShutdown(Xfile *xf);
EXPORT int XfileFprint(Xfile *xf, FILE *iStream);
EXPORT int XfileFdOpen(Xfile *xf, int iFdFlags, int iFdMode);
EXPORT void XfileFdClose(Xfile *xf);
EXPORT size_t XfileFdWrite(Xfile *xf, void *p, size_t iSize);
EXPORT off_t XfileFdSeekSet(Xfile *xf, off_t iOffset);
EXPORT off_t XfileFdSeekEnd(Xfile *xf, off_t iOffset);
#if __USE_LARGEFILE64
EXPORT off64_t XfileFdSeekSet64(Xfile *xf, off64_t iOffset);
EXPORT off64_t XfileFdSeekEnd64(Xfile *xf, off64_t iOffset);
#endif
EXPORT int XfileExists(Xfile *xf);
EXPORT size_t XfileLength(Xfile *xf);
EXPORT void XfileGrowBy(Xfile *xf, size_t iWlen);
EXPORT void XfileGrowToParts(Xfile *xf, size_t iSize1, size_t iSize2);

// Xmap
EXPORT Xmap *XmapNew(const char *iPathname);
EXPORT Xmap *XmapNewXfile(Xfile *xf);
EXPORT void XmapInit(Xmap *xm);
EXPORT void XmapDelete(Xmap *xm);
EXPORT void XmapDeleteXfile(Xmap *xm);
EXPORT void XmapUnmap(Xmap *xm);
#if _GNU_SOURCE
EXPORT void XmapRemap(Xmap *xm, size_t iLen);
#endif
EXPORT void XmapShutdown(Xmap *xm);
EXPORT void XmapDelete(Xmap *xm);
EXPORT int XmapFprint(Xmap *xm, FILE *iStream);

EXPORT void *XmapFile(Xmap * xm, off_t iOffset, size_t iLen, int iProt);
#if __USE_LARGEFILE64
EXPORT void *XmapFile64(Xmap * xm, off64_t iOffset, size_t iLen, int iProt);
#endif
EXPORT void *XmapOpenRO(Xmap * xm, off_t iOffset, size_t iLen);
EXPORT void *XmapOpenRW(Xmap * xm, off_t iOffset, size_t iLen);
EXPORT void XfileModeSpecs(XfileMode iMode, int *oFlags, int *oMode, int *oProt);
EXPORT void *XmapOpen(Xmap * xm, off_t iOffset, size_t iLen, XfileMode iMode);
EXPORT void *XmapOpenAllRO(Xmap * xm);
EXPORT void *XmapOpenAllRW(Xmap * xm);
EXPORT void *XmapOpenAll(Xmap * xm, XfileMode iMode);
#if __USE_LARGEFILE64
EXPORT void *XmapOpenRead64(Xmap *xm, off64_t iOffset, size_t iLen);
EXPORT void *XmapOpenWrite64(Xmap *xm, off64_t iOffset, size_t iLen);
EXPORT void *XmapOpen64(Xmap *xm, off64_t iOffset, size_t iLen, XfileMode iMode);
#endif
EXPORT void *XmapCreate(Xmap *xm, off_t iOffset, size_t iLen);
#if __USE_LARGEFILE64
EXPORT void *XmapCreate64(Xmap *xm, off64_t iOffset, size_t iLen);
#endif

EXPORT int xstrleft(const char * iStr, const char * iPrefix);
EXPORT const char * xstrskip(const char * iStr, const char * iSkip);
// Return true if file exists
EXPORT int xisfile(const char * iPathname);
EXPORT double xtimesec(struct timespec iFromTime);
// Setup desired state of the floating point subsystem
EXPORT void xfpsetup(void);

EXPORT int IsUnderDebugger(void);

#if __cplusplus
}
#endif

#define CSUMADD(a,b) (((a)<<=1)^=(b))

//Scaffold generated for every get/set/print/scan-able class field/member. TODO HOWTO template this? Need:
//  1. Macro -> template.
//  2. Append every new field to lists automagically. Templatable with comptime matching/recursion, TBD once #1 is solved.
//  3. Add private / public argument and let the macros define the field as well?
//Until figured out, do everything manually - doh.
//Example use:
//  FIELD_DEFINE(private, double, MinLogWeight, "%lg");
#define FIELD_DEFINE(scope, type, name, spec) \
public: \
  type Set ## name(type i ## name) { type tmp = m ## name; m ## name = i ## name; return tmp; } \
  type Get ## name(void) { return m ## name; } \
  void Print ## name(FILE *f) { fprintf(f, #name ": " spec "\n", m ## name); } \
  int Scan ## name(FILE *f) { return fscanf(f, #name ": " spec , & m ## name); } \
scope: \
  type m ## name

#define FIELD_SET(dstptr, name) (dstptr)->Set ## name(m ## name)
#define FIELD_GET(dstptr, name) (dstptr)->Get ## name()
#define FIELD_PRINT(f, name) Print ## name(f)
#define FIELD_SCAN(f, name) Scan ## name(f)

#endif

