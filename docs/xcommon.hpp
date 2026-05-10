//
// Common functionality for C++ programs that will not be included into C programs. Assume it will be compiled as a C++, not C.
// Templated functions in A.hpp, at link time appear in the object B.o off B.cpp (that #include-d A.hpp), *not* in A.o off possibly A.cpp.
//
// Generate C++ tags (alt: $ export CTAGS='-R --c++-kinds=+p --fields=+iaS --extra=+q' then $ ctags .): $ ctags -R --c++-kinds=+p --fields=+iaS --extra=+q {dirs}.
// Generate C++ tags using next gen https://github.com/universal-ctags/ctags ctags: $ /usr/local/bin/ctags -R --c++-kinds=+p --fields=+iaS --extras=+q --links=no {dirs}.
//
// $Id: $

#ifndef XCOMMON_HPP
#define XCOMMON_HPP

/* The includes needed by xcommon.hpp are:

#include <cmath>
#include <climits> // INT_MAX
#include <bsd/string.h> // strlcpy, needs -lbsd too; needs on ubuntu $ apt install libbsd-dev, on centos $ yum install libbsd-devel
#include <cstdarg> // va_list
#include <ctime> // time
#include <algorithm>
#include <string>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "xcommon.h"

*/

// Produce the class name at compile time (requires C++11).
// https://stackoverflow.com/questions/1666802/is-there-a-class-macro-in-c
// NB it is not a string & relies on "this" => does not work in static functions.
#define __CLASS__ std::remove_reference<decltype(classMacroImpl(this))>::type
template<class T> T& classMacroImpl(const T *t);

// Per class verbose output. Must be defined, default VERB0 to switch off.
// TODO FIXME Seems VERB(L, does not work here, but does work in xcommon.c for XALLOC_VERB - difference C v.s C++?
#define AUTOPTR_VERB(x) VERB0(("    AUTOPTR:%s:%d: ", __FILE__, __LINE__)); VERB0(x)

// Helper, otherwise AutoPtr return by value does not work. I.e making AutoPtr(const AutoPtr &)
// private not only disables passing by reference, but also returning by value?!
// Therefore when returning by value need to convert AutoPtr -> AutoPtrRef -> AutoPtr to get arround that.
template <class T> struct AutoPtrRef {

  AutoPtrRef(T *iPtr) : mPtr(iPtr) {
    AUTOPTR_VERB(("AutoPtrRef(T *) constructor %p now owns %p\n", (void *)this, (void *)mPtr));
  }
  AutoPtrRef() : mPtr(0) {
    AUTOPTR_VERB(("AutoPtrRef() default constructor %p\n", (void *)this));
  }

  // The copy constructor can not be defined. Results in error on instantiation in a context where AutoPtrRef is to be
  // used - when AutoPtr is returned by value and the conversion trick AutoPtr -> AutoPtrRef -> AutoPtr happens.
  //AutoPtrRef(AutoPtrRef<T> & iApr) {
  //  AUTOPTR_VERB(("AutoPtrRef(AutoPtrRef &) %p copy constructor owns %p called with %p owning %p but will not do it\n",
  //    (void *)this, (void *)mPtr, (void *)(& iApr), (void *)iApr.mPtr));
  //}

  void operator=(AutoPtrRef<T> & iApr) {
    AUTOPTR_VERB(("AutoPtrRef void operator=(AutoPtrRef &) %p equals operator owns %p called to equal %p that "
      "owns %p but will not do it!\n", (void *)this, (void *)mPtr, (void *)(& iApr), (void *)iApr.mPtr));
  }

  T *Release(void) {
    AUTOPTR_VERB(("AutoPtrRef T *Release() %p releasing %p\n", (void *)this, (void *)mPtr));
    T *p = mPtr;
    mPtr = nullptr;
    return p;
  }

  ~AutoPtrRef() {
    AUTOPTR_VERB(("~AutoPtrRef() destructor %p will NOT delete %p\n", (void *)this, (void *)mPtr));
  }

  T *mPtr;
};

// Canibalized SPtr from www.relisoft.com, changed as needed
template <class T> class AutoPtr {

public:

  // Default constructor
  AutoPtr(): mPtr(nullptr) {
    AUTOPTR_VERB(("AutoPtr() default constructor %p own nothing\n", (void *)this));
  }

  // Initialize from raw pointer
  explicit AutoPtr(T *iPtr) {
    AUTOPTR_VERB(("explicit AutoPtr(T *) %p initialise from raw pointer %p will forget/orphan existing %p\n",
      (void *)this, (void *)iPtr, (void *)mPtr));
    mPtr = iPtr;
  }

  // Copy constructor is ownership transfer
  AutoPtr(AutoPtr<T> & iAutoPtr) {
    AUTOPTR_VERB(("AutoPtr(AutoPtr &) copy constructor %p owns %p rhs is %p owns %p; will orphan %p and take over %p\n",
      (void *)this, (void *)mPtr, (void *)(& iAutoPtr), (void *)iAutoPtr.mPtr, (void *)iAutoPtr.mPtr, (void *)mPtr));
    mPtr = iAutoPtr.Release();
  }

  // Destroy current + ownership transfer
  void operator=(AutoPtr<T> & iAutoPtr) {
    AUTOPTR_VERB(("AutoPtr void operator=(AutoPtr &) %p equals operator owns %p", (void *)this, (void *)mPtr));
    if (mPtr != iAutoPtr.mPtr) {
      delete mPtr;
      mPtr = iAutoPtr.Release();
      AUTOPTR_VERB((" deleted now and acquired %p\n", (void *)mPtr));
    } else {
      AUTOPTR_VERB((" skipped assign to self\n"));
    }
  }

  // Can't really make the above with explicit names, as these "special" names constructors are used when
  // passing args by value etc. Ownership transfer + conversion. U must extend T to compile.
  template <class U> AutoPtr(AutoPtr<U> & up): mPtr(up.Release()) {
    AUTOPTR_VERB(("AutoPtr(AutoPtr<U> &) %p conversion from base U to T "
      "extending U now owns %p\n", (void *)this, (void *)mPtr));
  }

  // The payout - REMEMBER, no array[] for the underling, as NO delete[]
  ~AutoPtr() {
    AUTOPTR_VERB(("~AutoPtr() destructor %p will delete %p\n", (void *)this, (void *)mPtr));
    delete mPtr;
  }

  // Mimic raw pointer operators
  T *operator->() { return mPtr; }
  const T *operator->() const { return mPtr; }
  T & operator*() const { return *mPtr; }

  // Save & * effort
  T *RawPtr(void) const { return mPtr; }

  // Release ownership
  T *Release(void) {
    AUTOPTR_VERB(("AutoPtr T *Release() %p releasing %p\n", (void *)this, (void *)mPtr));
    T *tmp = mPtr;
    mPtr = nullptr;
    return tmp;
  }

  // Dangerous to call it operator=, not explicit enough. May be called in error, missing dereference etc
  //void operator=(T *iPtr) { Acquire(iPtr); }

  // Take ownership of raw pointer explicitly
  void Acquire(T *iPtr) {
    AUTOPTR_VERB(("AutoPtr Acquire(T *) %p will delete %p then own %p\n", (void *)this, (void *)mPtr, (void *)iPtr));
    if (mPtr != iPtr) delete mPtr; // delete 0 pointer is ok, delete checks => no need for us to check
    mPtr = iPtr;
  }

  // Make return by value possible now that the copy constructor is private. Doing Acquire(apr.mPtr) will make
  // us delete T that was never alloced. Seems mPtr is set to random stuff when "AutoPtr(AutoPtrRef) conversion"
  // is called? So the Acquire on apr.mPtr made us delete apr.mPtr but for that T *apr.mPtr the constructor was
  // never called? Therefore change the Acquire into Release and see what happens. TODO find documentation
  // that spells out explicitly what gets called when passing user defined types by value, reference, etc.
  AutoPtr(AutoPtrRef<T> iApr) {
    AUTOPTR_VERB(("AutoPtr(AutoPtrRef) conversion AutoPtr %p owns %p AutoPtrRef %p owns %p; AutoPtr %p will takeover %p "
      "from AutoPtrRef %p and forget about %p\n", (void *)this, (void *)mPtr, (void *)(& iApr), (void *)iApr.mPtr,
      (void *)this, (void *)iApr.mPtr, (void *)(& iApr), (void *)mPtr));
    mPtr = iApr.Release();
  }

  AutoPtr & operator=(AutoPtrRef<T> iApr) {
    AUTOPTR_VERB(("AutoPtr & operator=(AutoPtrRef) equals operator AutoPtr %p owns %p AutoPtrRef %p owns %p; "
      "AutoPtr %p will acquire %p from AutoPtrRef %p and delete %p\n", (void *)this, (void *)mPtr,
      (void *)(& iApr), (void *)iApr.mPtr, (void *)this, (void *)iApr.mPtr, (void *)(& iApr), (void *)mPtr));
    Acquire(iApr.mPtr);
    return *this;
  }

  template <class U> operator AutoPtrRef<U>() {
    AUTOPTR_VERB(("operator AutoPtrRef<U>() cast %p AutoPtr -> AutoPtrRef owned %p but will release now\n",
      (void *)this, (void *)mPtr));
    return AutoPtrRef<U>(Release());
  }

  template <class U> operator AutoPtr<U>() {
    AUTOPTR_VERB(("operator AutoPtr<U>() cast %p AutoPtr<T> -> AutoPtr<U> owned %p but will release now\n",
      (void *)this, (void *)mPtr));
    return AutoPtr<U>(Release());
  }

protected:

  T *mPtr;
};

// Don't pollute the global user namespace
#ifdef AUTOPTR_VERB
  #undef AUTOPTR_VERB
#endif

// Per class verbose output. Must be defined, default VEBR0 to switch off
#define POOL_VERB(x) VERB0(("    POOL:%s:%d: ", __FILE__, __LINE__)); VERB0(x)

// Pool of objects ("chunks") in blocks, atm returns memory back only when destroyed, there is not free-ing.
// Things to change/think about/cause concern:
// (a) Have Block and Chunk separate from Pool?
// (b) Have Pool::mSize template parameter => all the dynamic uglinees due to Block { Chunk ch[mSize] } will disappear.
//     Cons: not dynamic. But atm the "dynamicicity" is unused, constructor fixes mSize(xxx) first thing?
// (c) Rename to Xpool in line with rest of xcommon.c. But atm is in sync with AutoPtr which is not Xptr.
//
template <class T> class Pool {

public:

  // Default constructor
  Pool() : mSize(1003), mMaxBlocks(0), mNumBlocks(0), mHead(nullptr), mFree(nullptr) {
    POOL_VERB(("Pool() %p sizeof(T) %u siz %d\n", this, sizeof(T), mSize));
  }
  // User defined chunks per block
  Pool(int iSize) : mSize(iSize), mMaxBlocks(0), mNumBlocks(0), mHead(nullptr), mFree(nullptr) {
    POOL_VERB(("Pool() %p sizeof(T) %u siz %d\n", this, sizeof(T), mSize));
    XASSERT(mSize > 0);
  }
  // Turn on limit on number of blocks
  Pool(int iSize, int iMaxBlocks) : mSize(iSize), mMaxBlocks(iMaxBlocks), mNumBlocks(0), mHead(nullptr), mFree(nullptr) {
    POOL_VERB(("Pool() %p sizeof(T) %u siz %d maxblk %d\n", this, sizeof(T), mSize, mMaxBlocks));
    XASSERT(mSize > 0);
  }
  // Destructor, free the linked list of blocks
  ~Pool() {
    while (mHead) {
      Block *b = mHead->prev;
      delete[] mHead;
      mHead = b;
    }
  }
  // Allocate chunk
  T *Alloc(void) {
    XASSERT(Valid(INT_MAX));
    // If no free chunks in block get new block
    if (! mFree) {
        // May fail, if limited
        if (! AllocBlock()) return nullptr;
    }
    // Unllink chunk mFree points to, point mFree to next free or zero if no more free hunks
    Chunk *t = mFree;
    mFree = nullptr;
    if (t->l.prev) {
      XASSERT(t->l.prev->l.next == t);
      if (! mFree) mFree = t->l.prev;
      t->l.prev->l.next = t->l.next;
    }
    if (t->l.next) {
      XASSERT(t->l.next->l.prev == t);
      if (! mFree) mFree = t->l.next;
      t->l.next->l.prev = t->l.prev;
    }
    XASSERT(Valid(INT_MAX));
    T *const p = & t->t;
    // Should be busy chunk at this point
    XASSERT(ValidBusyT(p));
    return p;
  }

  // Free chunk
  void Free(T *p) {
    // Assert null ptr in debug, ignore otherwise
    XASSERT(p);
    if (! p) return;

    XASSERT(Valid(INT_MAX));
    XASSERT(ValidBusyT(p));

    // Must have at least a block allocated
    XASSERT(mHead);

    // Linkin chunk as first
    Chunk *c = (Chunk *)p;
    if (mFree) {
      Chunk *tn = mFree; // will become next
      Chunk *tp = mFree->l.prev; // will become prev
      // Linkin prev
      if (tp) {
        XASSERT(tp->l.next == tn);
        tp->l.next = c;
      }
      c->l.prev = tp;

      // Linkin next
      if (tn) {
        XASSERT(tn->l.prev == tp);
        tn->l.prev = c;
      }
      c->l.next = tn;
    } else {
      // This is the first free chunk
      c->l.prev = nullptr;
      c->l.next = nullptr;
    }
#if DEBUG
    c->l.user = -1; // counter in debug mode - set to "recycled"
#endif
    // Safe to linkin now
    mFree = c;

    // Should be free chunk at this point
    XASSERT(ValidFreeT(p));
    XASSERT(Valid(INT_MAX));
  }

private:

  // Stores one unit of data
  union Chunk {
    // When occupied, user value
    T t;
    // When free, chains free chunks in doubly linked list
    struct {
      Chunk *prev;
      Chunk *next;
#if DEBUG
      int user;
#endif
    } l;
  };

  // Stores mSize units of data
  struct Block {
    Block *prev;
#if DEBUG
    int user;
#endif
    Chunk ch[1]; // variable, [mSize]
  };

  int SizeOfBlock(int iSize) {
    return sizeof(Block) + (iSize - 1) * sizeof(Chunk);
  }

#if DEBUG
  // Validate the structure, level l=INT_MAX highest, 0 none
  int Valid(int l, ...) {
    // Level 1 tests - (almost) always performed
    if (1 <= l) {
      // Ensure blocks within limit or unlimited
      XASSERT((mMaxBlocks == 0 || mNumBlocks <= mMaxBlocks));
      // Block user data counter must be incrementing, starting with 0
      for (Block *b = mHead; b; b = b->prev) {
        XASSERT((b->prev==nullptr && b->user==0) || (b->user-1 == b->prev->user));
        }
    }
    // Level 2 tests - more time consuming
    if (2 <= l) {
    }
    // Higher levels - even more time consuming etc
    if (3 <= l) {
    }
    return 1;
  }

  // Ensure t points to non-free slot
  int ValidBusyT(const T *t) {
    // Ensure points in some block
    XASSERT(getInBlockCount(t) == 1);
    // Ensure points to non-free
    XASSERT(getFreeWalkCount(t) == 0);
    return 1;
  }

  // Ensure t points to free slot
  int ValidFreeT(const T *t) {
    // Ensure points in some block
    XASSERT(getInBlockCount(t) == 1);
    // Ensure points to free
    XASSERT(getFreeWalkCount(t) == 1);
    return 1;
  }

  // Walk the free chunks list, count number of appearances
  int getFreeWalkCount(const T *t) {
    // Walk both sides of the free list
    int inFree = 0;
    // Walk right
    for (Chunk *c = mFree; c; c = c->l.next) {
      if (c == (Chunk *)t) ++inFree;
    }
    // Walk left, don't double count mFree
    if (mFree) {
      for (Chunk *c = mFree->l.prev; c; c = c->l.prev) {
        if (c == (Chunk *)t) ++inFree;
      }
    }
    return inFree;
  }

  // Count t pointing into within a block
  int getInBlockCount(const T *t) {
    // Check it points in allocated block
    int inBlock = 0;
    for (Block *b = mHead; b; b = b->prev) {
      if ((void *)b <= (void *)t && (void *)t < (char *)b + SizeOfBlock(mSize)) {
        ++inBlock;
        // Don't break, count all
      }
    }
    return inBlock;
  }
#else
  // Will not be used, but needed in release mode otherwise XASSERT(Valid()) calls trigger:
  // error: there are no arguments to ‘Valid’ that depend on a template parameter, so a declaration of ‘Valid’ must be available [-fpermissive]
  int Valid(int l, ...) { return 0; }
#endif


  // Insert new block at head, fail if limited and at limit.
  // Returns succeed/failed flag.
  int AllocBlock(void) {
    XASSERT(Valid(INT_MAX));
    // Fail if at limit
    if (mMaxBlocks > 0 && !(mNumBlocks < mMaxBlocks)) return 0;

    // Save
    Block *const b = mHead;

    // Variable lenght mSize Block, may fail
    // FIXME this is wrong, does not guarantee that the Chunks in Chunk[[mSize] in this block are
    // alligned on 4 or 8 or whatever boundary! Only new Chunk[mSize] can guarantee this.
    Block *const b1 = (Block *)new char[SizeOfBlock(mSize)];
    if (! b1) return 0;

    // Linkin new block
    mHead = b1;
    if (mMaxBlocks > 0) ++mNumBlocks;
#if DEBUG
    mHead->user = (b? b->user+1: 0); // counter in debug mode
#endif
    // Setup the free list in the new block, with zero terminators both sides
    for (int i = 0; i < mSize; i++) {
      mHead->ch[i].l.prev = (i>0?  & mHead->ch[i-1]: 0);
      mHead->ch[i].l.next = (i<mSize-1? & mHead->ch[i+1]: 0);
#if DEBUG
      mHead->ch[i].l.user = i; // counter in debug mode
#endif
    }
    // Insert the entire list in the free, if any free
    if (mFree) {
      mHead->ch[mSize-1].l.next = mFree;
      Chunk *fp = mFree->l.prev;
      mFree->l.prev = & mHead->ch[mSize-1];

      fp->l.next = & mHead->ch[0];
      mHead->ch[0].l.prev = fp;
    } else {
      // Link to the first chunk if no existing free
      mFree = & mHead->ch[0];
    }

    // Linkin the new block
    mHead->prev = b;

    XASSERT(Valid(INT_MAX));

    return 1;
  }

  // Free the block at head
  void FreeBlock(void) {
    if (mHead) {
      Block *b = mHead->prev;
      delete[] mHead;
      mHead = b;
      // Update blocks counter if limited
      if (mMaxBlocks > 0) {
        --mNumBlocks;
        XASSERT((mNumBlocks >= 0));
        XASSERT((mNumBlocks <= mMaxBlocks));
      }
    }
  }

  // Number of T items per block
  int mSize;

  // Max blocks allowed, 0 for unlimited
  int mMaxBlocks;
  // Number of blocks alloced - only when mMaxBlocks>0
  int mNumBlocks;

  // Points to the head block, singly linked list is terminated by 0
  Block *mHead;

  // Points to a free chunk, can point anywhere in the doubly linked list.
  // The list is terminated by 0 pointers either side.
  Chunk *mFree;

};

// Don't pollute the global user namespace
#ifdef POOL_VERB
  #undef POOL_VERB
#endif

// Optimize out operations for special cases like power of 2 sizes etc.
// Checked, these are resolved at compile time if S is const at compile time (e.g sizeof(T)).
// Operations on consts are further resolved into const result. Operations on vars are correctly
// inlined with gcc -O3, there is no function call.
//
// Example:
//  int result = OptOpsSiz<sizeof(T),int>::Div(int_intput)
//  int result = OptOpsSiz<sizeof(T),int>::Mod(some_int)
//
// template <unsigned S, class U> struct OptOpsSiz {
//     static inline U Div(U a) { return a / S; }
//     static inline U Mod(U a) { return a % S; }
// };
// template <class U> struct OptOpsSiz<4U,U> {
//     static inline U Div(U a) { return a >> 2; }
//     static inline U Mod(U a) { return a & 3; }
// };
// template <class U> struct OptOpsSiz<8U,U> {
//     static inline U Div(U a) { return a >> 3; }
//     static inline U Mod(U a) { return a & 7; }
// };
// template <class U> struct OptOpsSiz<16U,U> {
//     static inline U Div(U a) { return a >> 4; }
//     static inline U Mod(U a) { return a & 15; }
// };
//
// Annoying to have to spell the type of the argument in the template params. So same thing, only as
// bunch of global functions, so compiler will pick the right on from the argument type itself. Silly can't pick the class
// that way? Add new sizes and types to the list.
// Defined the fallbacks or leave them undefined (or ill-define them) so evaluation to the fallback case causes compile
// time error. By defaut fallback.
#ifndef XOPT_OPS_FOR_TYPE_FAIL_ON_FALLBACK
  #define XOPT_OPS_FOR_TYPE_FAIL_ON_FALLBACK 1
#endif

#if XOPT_OPS_FOR_TYPE_FAIL_ON_FALLBACK
  // Case where fallbacks are defined to fail. Can't produce compile time error, because needs to compile even when
  // fallback is not used.
  //
  // Idea: change signature to return void, so to compile, but still fail to match at compile time. But, at compile time
  // need the general funcion declared for the specializations to be able to compile. And once declared, then that
  // introduces ambiguity between that one and the one returning void which is supposed to generate compile type
  // error. So same "polymorfism ignores the return type" that tought can be used for, actually makes it impossible
  // to use with. Do it the pedestrian way, with run-time fail.
  //
  #define USAGE(x,y) ("T ::"#x"<%u>(T) failed. You are using default non-optimised version (fallback) with XOPT_OPS_FOR_TYPE_FAIL_ON_FALLBACK=%d. Advice: (a) (Re)#define it to 0 and recompile to use non-failing slow fallback; (b) Fix the callee to make use of optimised version (c) Supply optimised version for your type T via "#y"(T)\n", S, XOPT_OPS_FOR_TYPE_FAIL_ON_FALLBACK)
  #define DEFINE_XOPT_OPS_FALLBACK_MUL_FOR_TYPE(T) \
  template <unsigned S> inline T XoptMul(T a) { XCHECK(0,USAGE(XoptMul,DEFINE_XOPT_OPS_FALLBACK_MUL_FOR_TYPE)); return a; }
  #define DEFINE_XOPT_OPS_FALLBACK_DIV_FOR_TYPE(T) \
  template <unsigned S> inline T XoptDiv(T a) { XCHECK(0,USAGE(XoptDiv,DEFINE_XOPT_OPS_FALLBACK_DIV_FOR_TYPE)); return a; }
  #define DEFINE_XOPT_OPS_FALLBACK_MOD_FOR_TYPE(T) \
  template <unsigned S> inline T XoptMod(T a) { XCHECK(0,USAGE(XoptMod,DEFINE_XOPT_OPS_FALLBACK_MOD_FOR_TYPE)); return a; }
  #define DEFINE_XOPT_OPS_FALLBACK_BLOCKUP_FOR_TYPE(T) \
  template <unsigned S> inline T XoptBlockUp(T a) { XCHECK(0,USAGE(XoptBlockUp,DEFINE_XOPT_OPS_FALLBACK_BLOCKUP_FOR_TYPE)); return a; }
  #define DEFINE_XOPT_OPS_FALLBACK_BLOCKUPTO_FOR_TYPE(T) \
  template <unsigned S> inline T XoptBlockUpTo(T a, T b) { XCHECK(0,USAGE(XoptBlockUpTo,DEFINE_XOPT_OPS_FALLBACK_BLOCKUPTO_FOR_TYPE)); return a; }
  #define DEFINE_XOPT_OPS_FALLBACK_ROUNDUP_FOR_TYPE(T) \
  template <unsigned S> inline T XoptRoundUp(T a) { XCHECK(0,USAGE(XoptRoundUp,DEFINE_XOPT_OPS_FALLBACK_ROUNDUP_FOR_TYPE)); return a; }
#else
  // With slow fallacks, don't error at run time, use the slow (fallback) verions
  #define DEFINE_XOPT_OPS_FALLBACK_MUL_FOR_TYPE(T) \
  template <unsigned S> inline T XoptMul(T a) { return a * S; }
  #define DEFINE_XOPT_OPS_FALLBACK_DIV_FOR_TYPE(T) \
  template <unsigned S> inline T XoptDiv(T a) { return a / S; }
  #define DEFINE_XOPT_OPS_FALLBACK_MOD_FOR_TYPE(T) \
  template <unsigned S> inline T XoptMod(T a) { return a % S; }
  #define DEFINE_XOPT_OPS_FALLBACK_BLOCKUP_FOR_TYPE(T) \
  template <unsigned S> inline T XoptBlockUp(T a) { return (a%S? a/S+1: a/S); }
  #define DEFINE_XOPT_OPS_FALLBACK_BLOCKUPTO_FOR_TYPE(T) \
  template <unsigned S> inline T XoptBlockUpTo(T a, T b) { const T b1=(a%S? a/S+1: a/S); return (b1>b? b: b1); }
  #define DEFINE_XOPT_OPS_FALLBACK_ROUNDUP_FOR_TYPE(T) \
  template <unsigned S> inline T XoptRoundUp(T a) { return (a%S? S*(a/S+1): a); }
#endif

// Too bad ctags does not grok this macro expansion. Is it safe (i.e will compiler still inline) to
// express BlockUp in terms of simpler functions (Mod,Div) or should one stick to "no function
// calls, plain expressions only"? Stick to no function calls until too painful.
#define DEFINE_XOPT_OPS_FOR_TYPE(T) \
  DEFINE_XOPT_OPS_FALLBACK_MUL_FOR_TYPE(T) \
  template <> inline T XoptMul<2U>(T a) { return a << 1; } \
  template <> inline T XoptMul<4U>(T a) { return a << 2; } \
  template <> inline T XoptMul<8U>(T a) { return a << 3; } \
  template <> inline T XoptMul<16U>(T a) { return a << 4; } \
  DEFINE_XOPT_OPS_FALLBACK_DIV_FOR_TYPE(T) \
  template <> inline T XoptDiv<2U>(T a) { return a >> 1; } \
  template <> inline T XoptDiv<4U>(T a) { return a >> 2; } \
  template <> inline T XoptDiv<8U>(T a) { return a >> 3; } \
  template <> inline T XoptDiv<16U>(T a) { return a >> 4; } \
  DEFINE_XOPT_OPS_FALLBACK_MOD_FOR_TYPE(T) \
  template <> inline T XoptMod<2U>(T a) { return a & 1; } \
  template <> inline T XoptMod<4U>(T a) { return a & 3; } \
  template <> inline T XoptMod<8U>(T a) { return a & 7; } \
  template <> inline T XoptMod<16U>(T a) { return a & 15; } \
  DEFINE_XOPT_OPS_FALLBACK_BLOCKUP_FOR_TYPE(T) \
  template <> inline T XoptBlockUp<2U>(T a) { return (a&1? (a>>1)+1: a>>1); } \
  template <> inline T XoptBlockUp<4U>(T a) { return (a&3? (a>>2)+1: a>>2); } \
  template <> inline T XoptBlockUp<8U>(T a) { return (a&7? (a>>3)+1: a>>3); } \
  template <> inline T XoptBlockUp<16U>(T a) { return (a&15? (a>>4)+1: a>>4); } \
  DEFINE_XOPT_OPS_FALLBACK_BLOCKUPTO_FOR_TYPE(T) \
  template <> inline T XoptBlockUpTo<2U>(T a, T b) { const T b1=(a&1? (a>>1)+1: a>>1); return (b1>b? b: b1); } \
  template <> inline T XoptBlockUpTo<4U>(T a, T b) { const T b1=(a&3? (a>>2)+1: a>>2); return (b1>b? b: b1); } \
  template <> inline T XoptBlockUpTo<8U>(T a, T b) { const T b1=(a&7? (a>>3)+1: a>>3); return (b1>b? b: b1); } \
  template <> inline T XoptBlockUpTo<16U>(T a, T b) { const T b1=(a&15? (a>>4)+1: a>>4); return (b1>b? b: b1); } \
  DEFINE_XOPT_OPS_FALLBACK_ROUNDUP_FOR_TYPE(T) \
  template <> inline T XoptRoundUp<2U>(T a) { return (a&1? a+1: a); } \
  template <> inline T XoptRoundUp<4U>(T a) { return (a&3? a+4-(a&3): a); } \
  template <> inline T XoptRoundUp<8U>(T a) { return (a&7? a+8-(a&7): a); } \
  template <> inline T XoptRoundUp<16U>(T a) { return (a&15? a+16-(a&15): a); }

// Define the above operators for 6 types of interest
DEFINE_XOPT_OPS_FOR_TYPE(unsigned short)
DEFINE_XOPT_OPS_FOR_TYPE(short)
DEFINE_XOPT_OPS_FOR_TYPE(unsigned int)
DEFINE_XOPT_OPS_FOR_TYPE(int)
DEFINE_XOPT_OPS_FOR_TYPE(unsigned long)
DEFINE_XOPT_OPS_FOR_TYPE(long)

// Duplicates are not allowed, so DEFINE_XOPT_OPS_FOR_TYPE(size_t) is an error


// Use compile time known static buffer size (https://randomascii.wordpress.com/2013/04/03/stop-using-strncpy-already/):
// char buf[5]; xstrcpy(buf, "This is a long string");
template <size_t dstSize>
void xstrcpy(char (& oDst)[dstSize], const char *iSrc) {
  // Place here function that copies up to max dstSize bytes from iSrc to oDst
  #if 0
    // Use standard strncpy. Problems: 1) always copies dstSize bytes even if iSrc is much smaller; 2) not 100% certain it will put \0 at end.
    strncpy(oDst, iSrc, dstSize); // copy the string
    oDst[dstSize - 1] = 0; // ensure null-termination
  #else
    // Use BSD strlcpy. Problems: 1) not libc nor POSIX, it's in libbsd on linux so link with -lbsd, need to install libbsd-dev (ubuntu) libbsd-devel (centos) and -lbsd needs to come *after* the .o file that is using it
    strlcpy(oDst, iSrc, dstSize); // copy the string
  #endif
}

// Append to existing string in a buffer of size learger than the string
template <size_t dstSize>
void xstrcat(char (& oDst)[dstSize], const char *iSrc) {
  // User BSD strlcat. Problems:
  //  1) not libc nor POSIX, it's in libbsd on linux so link with -lbsd, need to install libbsd-dev (ubuntu) libbsd-devel (centos)
  //  2) if traversing oDst does not find \0 within the dstSize, then oDst is untouched left unterminated
  strlcat(oDst, iSrc, dstSize); // copy the string
}

// Do snprintf on a static buffer, ensure terminating zero.
// Return the buffer pointer so to be able to chain expressions.
template <size_t DST_SZ>
char *xsnprintf(char (& oDst)[DST_SZ], const char *iFmt, ...) {
  va_list args;
  va_start(args, iFmt);
  vsnprintf(oDst, DST_SZ, iFmt, args);
  va_end(args);
  oDst[DST_SZ - 1] = '\0';
  return oDst;
}


// As above but use number of chars in, and update on out. So return the number of chars from start of the string.
template <size_t DST_SZ>
char *xsnprintf(char (& oDst)[DST_SZ], int *ioCnt, const char *iFmt, ...) {
  XASSERT(0 <= (*ioCnt) && (*ioCnt) < (int)DST_SZ);
  if ((*ioCnt) < (int)DST_SZ) {
    va_list args;
    va_start(args, iFmt);
    *ioCnt += vsnprintf(oDst + (*ioCnt), DST_SZ - (*ioCnt), iFmt, args);
    va_end(args);
    oDst[DST_SZ - 1] = '\0';
  }
  return oDst;
}

// Concatenate snprintf on a static buffer, ensure terminating zero.
template <size_t DST_SZ>
char *xstrcatprintf(char (& oDst)[DST_SZ], const char *iFmt, ...) {
  int n = strlen(oDst);
  if (n > (int)DST_SZ - 2) return oDst;
  va_list args;
  va_start(args, iFmt);
  vsnprintf(oDst + n, DST_SZ - n, iFmt, args);
  va_end(args);
  oDst[DST_SZ - 1] = '\0';
  return oDst;
}

// Provide isnan() function per type.
// NB Template specializations must be inline in .hpp file, otherwise definition must go in a .cpp file.
template <typename T> bool isnan(T v);
template<> inline bool isnan<int>(int v) { return v == INT_NAN; }
template<> inline bool isnan<long>(long v) { return v == LONG_NAN; }
template<> inline bool isnan<long long>(long long v) { return v == LLONG_NAN; }
template<> inline bool isnan<float>(float v) { return std::isnan(v); }
template<> inline bool isnan<double>(double v) { return std::isnan(v); }

// Provide isinf() function per type.
// NB Template specializations must be inline in .hpp file, otherwise definition must go in a .cpp file.
template <typename T> bool isinf(T v);
template<> inline bool isinf<int>(int v) { return v == INT_INF; }
template<> inline bool isinf<long>(long v) { return v == LONG_INF; }
template<> inline bool isinf<long long>(long long v) { return v == LLONG_INF; }
template<> inline bool isinf<float>(float v) { return std::isinf(v); }
template<> inline bool isinf<double>(double v) { return std::isinf(v); }

// Function taking argument of type T, returns true/false
template <typename T> using isok_t = bool (*)(T);
template <typename T> bool isyes(T) { return true; }
template <typename T> bool ispos(T a) { return a>0; }
template <typename T> bool isposnum(T a) { return !isnan(a) && (a>0); }

// Find last <=, return the index, ignoring any element that is not ok, like NAN or othererwise as defined in the predicate
template <typename T> int find_last(const T *arr, int siz, T v, isok_t<T> iIsOk =  & isyes) {
  int j = INT_NAN;
#if XASSERT_IF
  T b;
  bool c = false;
#endif
  for (int i = 0; i < siz; ++i) {
    if (! (*iIsOk)(arr[i])) { continue; }
    XASSERT(!c || b <= arr[i]); // ensure the ok elements are sorted
    if (arr[i] <= v) { j = i; }
    if (arr[i] > v) { break; }
#if XASSERT_IF
    b = arr[i];
    c = true;
#endif
  }
  return j;
}

// Find first >=, return the index, ignoring any element that is not ok, like NAN or otherwise as defined in the predicate
template <typename T> int find_first(const T *arr, int siz, T v, isok_t<T> iIsOk =  & isyes) {
  int j = INT_NAN;
#if XASSERT_IF
  T b;
  bool c = false;
#endif
  for (int i = 0; i < siz; ++i) {
    if (! (*iIsOk)(arr[i])) { continue; }
    XASSERT(!c || b <= arr[i]); // assert the ok elements are sorted
    if (v >= arr[i]) {
      j = i;
      break;
    }
#if XASSERT_IF
    b = arr[i];
    c = true;
#endif
  }
  return j;
}

// Vector summary
template <size_t DST_SIZ, typename T>
char *xvecsummary(char (& oDst)[DST_SIZ], const T *iVec, long iN, const char *iFilename) {

  XASSERT(iN >= 0);

  // Compute easy stats - min, max, +ves, -ves, 0s, NaNs, Infs, mean, stdev
  float v_min = NAN;
  for (int i = 0; i < iN; ++i) {
    if (! isnan(iVec[i])) { // in extremis, will do one pass just for this
      v_min = iVec[i];
      break;
    }
  }
  float v_max = v_min;
  float v_sum = 0, v_sum2 = 0, v_mu = 0, v_sd = 0, v_sum_pos = 0, v_sum_neg = 0;
  long v_neg = 0, v_zer = 0, v_pos = 0, v_inf = 0, v_nan = 0, v_num = 0;
  for (long i = 0; i < iN; ++i) {
    const T v = iVec[i];
    if (isnan(v)) {
      ++v_nan;
    } else if (isinf(v)) {
      ++v_inf;
    } else if (v > 0) {
      ++v_pos;
      v_sum_pos += v;
    } else if (v == 0) {
      ++v_zer;
    } else if (v < 0) {
      ++v_neg;
      v_sum_neg += v;
    } else {
      XERROR(("Bad element #%ld value %g", i, (double)v));
    }
    if (!isnan(v) && !isinf(v)) {
      if (v < v_min) v_min = v;
      if (v > v_max) v_max = v;
      v_sum += v;
      v_sum2 += v * v;
      ++v_num;
    }
  }
  if (v_num > 0) {
    v_mu = v_sum / (float)v_num;
    v_sd = std::sqrt(v_sum2 / (float)v_num - v_mu * v_mu);
    // sd = SQRT(1/N * SUM_i((x_i - mu)^2)) = SQRT(1/N*SUM_i(x_i^2) - 1/N*2*mu*SUM_i(x_i) + 1/N*N*mu^2) = SQRT(1/N*SUM_i(x_i^2) - mu^2)
  }

  // Finding the quantiles is destructive => must make a copy
  if (iFilename) {
    // Turn on per-filename locking, or use global critical section. NB use #ifdef _OPENMP to check if compiling under -fopenmp.
#define FILENAME_LOCK 1
    float p0 = NAN, p1 = NAN, p5 = NAN, p25 = NAN, p50 = NAN, p75 = NAN, p95 = NAN, p99 = NAN, p100 = NAN;
    long pN = 0;
#if FILENAME_LOCK
    static std::unordered_map<std::string, omp_lock_t *> str2lock;
#endif

    // A full critical section is too strict. It is 1) only needed when iFilename is a genuione mmap, and then,
    // 2) only one lock per name. When iFilename="" there is not a need to lock as memory is allocated on heap.
    // So have an unorderered_map<char *,lock_t *>, and acquire/release a lock per mmap filename.
#if !FILENAME_LOCK
    #pragma omp critical (Vsummary)
#endif
    {
#if FILENAME_LOCK
      // This is leaked and is expected, but the leak detector at the exit of the program complains. 
      // NB clang allows for no_sanitize. NB gcc does not, emits a warning, disable it on this line only.
      // But the problem of the leak detector detecting this leak in gcc remains.
      // Double underscores to be safer aviod name conflict with an existing macro.
      // https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer
      // Alternative is to suppress the leak in this function only with a special text file:
      //    $ cat suppr.txt
      //    leak:xvecsummary
      // then run with
      //    ASAN_OPTIONS=detect_leaks=1 LSAN_OPTIONS=suppressions=suppr.txt ./a.out
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
      omp_lock_t *o __attribute__((__no_sanitize__("leak"))) = NULL;
#pragma GCC diagnostic pop
#endif
      // If no name was passed (pointer is NULL) => skip the vector summary.
      // If a valid (non-empty) name was passed => create a mmap-ed file to hold a temporary copy of the iVec input array.
      // If an empty name (string "") was passed => create the temp iVac RW copy on the heap.
      // The input iVec is RO, but for the quantiles calc we need an array of the same size that is RW.
      float *vec;
      Xmap *vecMap = 0;
      if (*iFilename) {
        // If multithreaded, multiple threads may try to use the same temp backing store => lock
#if FILENAME_LOCK
        auto p = str2lock.find(iFilename);
        if (p == str2lock.end()) {
          o = new omp_lock_t; // will never be freed
          omp_init_lock(o);
          str2lock[iFilename] = o;
        } else {
          o = p->second;
        }
        // This call either acquires the lock, or blocks and waits until it can be acquired
        omp_set_lock(o);
#endif
        size_t siz = sizeof(float) * iN;
        vecMap = XmapNew(iFilename);              // creates the map, but the underlying file maybe empty
        XmapOpenRW(vecMap, 0, siz);               // create new file or open existing for writing, map the [0,siz) region
        XfileGrowToParts(vecMap->mXfile, siz, 0); // grow the file to siz, or do nothing if already at size >= siz
        vec = (float *)(vecMap->mMap);            // setup data array to point to the file content
      } else {
        vec = XMALLOC(float, iN);
      }

      // Copy the input RO iVec array over to temp RW vec array, that will be partially sorted during std::nth_element.
      // Skip the zeros and the nans, use only non-zero numbers. It is both faster, more interesting, and we output the number of zeros too.
      float *vec_end = std::copy_if(& iVec[0], & iVec[iN], & vec[0], [](T v){return !isnan(v) && std::fabs((double)v)>1e-6; } );

      // The percentiles are computed with partial ordering of elements - no need for a full sort.
      // No need to re-copy from iVec on every nth_element too - they will get re-sorted as needed.
      pN = std::distance(vec, vec_end);
      if (pN > 0) {
        float *p;
        p = std::min_element(& vec[0], vec_end);                        p0  = *p;
        p = & vec[pN/100];      std::nth_element(& vec[0], p, vec_end); p1  = *p; XASSERT(p0  <= p1);
        p = & vec[pN/20];       std::nth_element(& vec[0], p, vec_end); p5  = *p; XASSERT(p1  <= p5);
        p = & vec[pN/4];        std::nth_element(& vec[0], p, vec_end); p25 = *p; XASSERT(p5  <= p25);
        p = & vec[pN/2];        std::nth_element(& vec[0], p, vec_end); p50 = *p; XASSERT(p25 <= p50);
        p = & vec[(3*pN)/4];    std::nth_element(& vec[0], p, vec_end); p75 = *p; XASSERT(p50 <= p75);
        p = & vec[(19*pN)/20];  std::nth_element(& vec[0], p, vec_end); p95 = *p; XASSERT(p75 <= p95);
        p = & vec[(99*pN)/100]; std::nth_element(& vec[0], p, vec_end); p99 = *p; XASSERT(p95 <= p99);
        p = std::max_element(& vec[0], vec_end);                        p100= *p; XASSERT(p99 <= p100);
      }

      // Not necessary, but free some address space jic, do not leave mmap-s dangling
      if (*iFilename) {
        XmapDelete(vecMap);
#if FILENAME_LOCK
        XASSERT(o);
        omp_unset_lock(o);
#endif
      } else {
#if FILENAME_LOCK
        XASSERT(! o);
#endif
        XFREE(vec);
      }
      snprintf(oDst, DST_SIZ, "nnz num %ld %.2f%% min %g p1 %g p5 %g p25 %g p50 %g p75 %g p95 %g p99 %g max %g all min %g max %g mu %g sd %g num %ld %.2f%% sumneg %g neg %ld %.2f%% zer %ld %.2f%% pos %ld %.2f%% sumpos %g num %ld %.2f%% inf %ld %.2f%% nan %ld %.2f%% siz %ld",
        pN, (float)pN / iN * 1e2, p0, p1, p5, p25, p50, p75, p95, p99, p100, v_min, v_max, v_mu, v_sd, v_num, (float)v_num / iN * 1e2, v_sum_neg, v_neg, (float)v_neg / iN * 1e2, v_zer, (float)v_zer / iN * 1e2, v_pos, (float)v_pos / iN * 1e2, v_sum_pos, iN - v_inf - v_nan, 100 - (float)(v_inf + v_nan) / iN * 1e2, v_inf, (float)v_inf / iN * 1e2, v_nan, (float)v_nan / iN * 1e2, iN);
    } // critical Vsummary
#undef FILENAME_LOCK

  } else {
    snprintf(oDst, DST_SIZ, "all nnz %ld %.2f%% inf %ld %.2f%% nan %ld %.2f%% num %ld %.2f%% min %g max %g mu %g sd %g neg %ld %.2f%% sumneg %g zer %ld %.2f%% pos %ld %.2f%% sumpos %g siz %ld",
    iN - v_zer - v_inf - v_nan, 100 - (float)(v_zer + v_inf + v_nan) / iN * 1e2, v_inf, (float)v_inf / iN * 1e2, v_nan, (float)v_nan / iN * 1e2, v_num, (float)v_num / iN * 1e2, v_min, v_max, v_mu, v_sd, v_neg, (float)v_neg / iN * 1e2, v_sum_neg, v_zer, (float)v_zer / iN * 1e2, v_pos, (float)v_pos / iN * 1e2, v_sum_pos, iN);
  }

  return oDst;
}

// Wrapper for the above, use [beg,end) instead of [vec,SIZ]
template <size_t DST_SIZ, typename T>
char *xvecsummary(char (& oDst)[DST_SIZ], const T *iVecBeg, const T *iVecEnd, const char *iFilename) {
  return xvecsummary<DST_SIZ, T>(oDst, iVecBeg, (long)(iVecEnd - iVecBeg), iFilename);
}

// Current time can be either clock_t t = clock(), or time_t t = time(nullptr), struct timespec t; clock_gettime(CLOCK_REALTIME, & t).
// Standardize on the clock_gettime().

// Current time as string into a buffer supplied by the user.
// Return the buffer pointer never the less, for chaining of expressions.
template <size_t DST_SZ>
char *xstrftime(char (& oDst)[DST_SZ]) {
  time_t t = time(nullptr);
  struct tm tm;
  localtime_r(& t, & tm);
  // The timezone part %Z does not work??
  strftime(oDst, DST_SZ, "%F %T %z", & tm);
  return oDst;
}

// Current time relative to prior time as string into a buffer supplied by the user.
// Better time delta measure via clock_gettime() instead of a simple clock(), if a bit less convenient?
// Assumes CLOCK_REALTIME. NB using CLOCK_MONOTONIC is fine for intervals but it is not the wall time.
template <size_t DST_SZ>
char *xstrftime(char (& oDst)[DST_SZ], struct timespec iFromTime) {
  double ts_delta = xtimesec(iFromTime);
  // Convert interval time in seconds to Nd HH:MM:DD
  time_t t = (time_t)ts_delta;
  struct tm tm;
  gmtime_r(& t, & tm);
  xsnprintf(oDst, "%dd %2.2d:%2.2d:%2.2d", tm.tm_yday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  return oDst;
}

// Current time relative to prior time and the ETA for Num work done in tange [Min,Max] as string into a buffer supplied by the user.
// Used in omp parallel loops, when iNow does not indicated the number of jobs done! The number of jobs done is separate iDone.
// Better time delta measure via clock_gettime() instead of a simple clock(), if a bit less convenient?
// Assumes CLOCK_REALTIME. NB using CLOCK_MONOTONIC is fine for intervals but it is not the wall time.
template <size_t DST_SZ>
char *xstrftime(char (& oDst)[DST_SZ], struct timespec iFromTime, int iFrom, int iNow, int iTo) {
  XASSERT(0 <= iFrom && iFrom <= iNow && iNow <= iTo);
  time_t t; // time_t is number of seconds since 1-Jan-1970
  struct tm tm;
  char buf1[33], buf2[33], buf3[33];
  // Get time elapsed in seconds
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  double ts_delta = (ts.tv_sec - iFromTime.tv_sec) + (ts.tv_nsec - iFromTime.tv_nsec) / 1000000000.0;
  // Convert interval time in seconds to HH:MM:DD
  t = (time_t)ts_delta;
  gmtime_r(& t, & tm); // would have though gmtime_r() here - but it looks like it adds the offset?
  xsnprintf(buf1, "%dd %2.2d:%2.2d:%2.2d", tm.tm_yday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  // Get ETA time - time remaining to end, extrapolated from how much elapsed from start
  t = (time_t)(ts_delta * ((double)(iTo - iNow) / (double)(iNow - iFrom + 1)));
  gmtime_r(& t, & tm);
  xsnprintf(buf2, "%dd %2.2d:%2.2d:%2.2d", tm.tm_yday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  // Get the current time too
  strftime(buf3, sizeof(buf3), "%F %T %z", localtime_r(& ts.tv_sec, & tm));
  // All together
  xsnprintf(oDst, "in %s %d/%d %.2f%% %d/[%d,%d] ETA %s at %s", buf1, iNow-iFrom, iTo-iFrom+1, (float)(iNow-iFrom)/(float)(iTo-iFrom+1)*1e2, iNow, iFrom, iTo, buf2, buf3);
  return oDst;
}

// Better time delta measure via clock_gettime() instead of a simple clock(), if a bit less convenient?
// Assumes CLOCK_REALTIME. NB using CLOCK_MONOTONIC is fine for intervals but it is not the wall time.
// Used in omp parallel loops, when iNow does not indicated the number of jobs done! The number of jobs done is separate iDone.
template <size_t DST_SZ>
char *xstrftime(char (& oDst)[DST_SZ], struct timespec iFromTime, int iFrom, int iNow, int iTo, int iDone) {
  XASSERT(0 <= iFrom && iFrom <= iNow && iNow <= iTo);
  XASSERT(0 <= iDone && iDone <= iTo - iFrom + 1);
  time_t t; // time_t is number of seconds since 1-Jan-1970
  struct tm tm;
  char buf1[33], buf2[33], buf3[33];
  // Get time elapsed in seconds
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  double ts_delta = (ts.tv_sec - iFromTime.tv_sec) + (ts.tv_nsec - iFromTime.tv_nsec) / 1000000000.0;
  // Convert interval time in seconds to HH:MM:DD
  t = (time_t)ts_delta;
  gmtime_r(& t, & tm); // not sure if localtime_r() or gmtime_r() here?
  xsnprintf(buf1, "%dd %2.2d:%2.2d:%2.2d", tm.tm_yday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  // Get ETA time - time remaining to end, extrapolated from how much elapsed from start
  t = (time_t)(ts_delta * ((double)(iTo - iFrom + 1 - iDone) / (double)iDone));
  gmtime_r(& t, & tm);
  xsnprintf(buf2, "%dd %2.2d:%2.2d:%2.2d", tm.tm_yday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  // Get the current time too
  strftime(buf3, sizeof(buf3), "%F %T %z", localtime_r(& ts.tv_sec, & tm));
  // All together
  xsnprintf(oDst, "in %s %d/%d %.2f%% %d/[%d,%d] ETA %s at %s", buf1, iDone, iTo - iFrom + 1, (float)iDone/(float)(iTo - iFrom + 1) * 1e2, iNow, iFrom, iTo, buf2, buf3);
  return oDst;
}

// Read one line into a static buffer, return true if ok.
// Use as: char buf[100]; xfreadline(buf,f); XCHECK(xfreadline(buf,f),("File %s can not read header",iPathname));
// If reading fails return false and put empty string in the buffer.
template <size_t DST_SZ>
int xfreadline(char (& oDst)[DST_SZ], FILE *f) {
  char *p = fgets(oDst, DST_SZ, f);
  if (p) {
    // If truncated will lack '\0', if read ok will have end of line '\n'
    oDst[DST_SZ - 1] = '\0';
    int k = strlen(oDst) - 1;
    if (oDst[k] == '\n') oDst[k] = '\0';
  } else {
    if (DST_SZ > 0) oDst[0] = '\0';
  }
  return p != nullptr;
}

// Number of elements of static array in compile time
template <typename T, size_t SZ> constexpr int XNUMEL(T (&arr)[SZ]) { return SZ; }

template <typename T, size_t SZ>
T & XFILL(T (& oDst)[SZ], const T & iVal) {
  std::fill(oDst, oDst + SZ, iVal);
  return *oDst;
}

// Wrapper arround the VerbXxx static functions and data. V:: static functions use the global static struct Verb sVerb.
// While the non-static V functions have their own buffer in mV, that may or may not be the same as global.
// The non-static V. functions can be applied to a different V buffer, and the V class object can be cloned and passed over.
// C++ does not allow overloading of the names on static yes/no only => prepend leading V to the non-static members.
//
#define XCOMMON_DEFINE_STATICS std::unordered_map<std::string_view, int> V::mStr2num
//
struct V {

  struct Verb mV = { 3, { '\0' } };

  // Copy from C++ struct V
  V(const V & iV) {
    #pragma omp critical (Verb)
    {
      mV.mLevel = iV.mV.mLevel;
      strcpy(& mV.mPrefix[0], & iV.mV.mPrefix[0]);
    }
  }
  // Copy from C struct Verb
  V(const struct Verb *iV) {
    #pragma omp critical (Verb)
    {
      mV.mLevel = iV->mLevel;
      strcpy(& mV.mPrefix[0], & iV->mPrefix[0]);
    }
  }

  // Get and set level
  int VL(void) { return VerbLevel2(& mV); }
  int VL(int v) { return VerbSet2(& mV, v); }

  // Is level or not?
  int VIS(int v) { return VerbIs2(& mV, v); }

  // Was keyword queried N times, or not? Specific to the C++, missing in the C Vereb class b/c no standard hash tables in C.
  // This hash table should stay static and be shared by all instances of struct V (and struct Verb, if needed).
  static std::unordered_map<std::string_view, int> mStr2num;
  int VIS(const char *s, int n) {
    int rc = 0;
    #pragma omp critical (Verb)
    {
      auto i = mStr2num.find(s);
      if (i == mStr2num.end()) {
        mStr2num[s] = 1;
        rc = 1;
      } else {
        int ii = i->second;
        if (ii < n) {
          mStr2num[s] = ii + 1;
          rc = 1;
        }
      }
    }
    return rc;
  }

  // Push and pop prefix - drop the P for brevity
  int VU(const char *iPrefix) { return VerbPush2(& mV, iPrefix); }
  int VO(void) { return VerbPop2(& mV); }

  // Print message if the verbosity level is high enough.
  // Copy of int Verb(int v, const char *iFmt, ... ), TODO unify.
  // Specific to gcc __attribute__ tells gcc the function takes vararg parameters that are printf format specifiers.
  int VP(int v, const char *iFmt, ... ) __attribute__((format(printf, 3, 4))) {
    int rc = 0;
    #pragma omp critical (Verb)
    {
      if (VerbIs2(& mV, v) && iFmt) {
        fflush(stdout); fflush(stderr);
        if (*VerbPrefix2(& mV)) printf("%s: ", VerbPrefix2(& mV));
        va_list list;
        va_start(list, iFmt);
        rc = vprintf(iFmt, list);
        va_end(list);
        fflush(stdout);
      }
    }
    return rc;
  }

  // Print message - always.
  // Copy of int Verbp(const char *iFmt, ... ) TODO unify.
  // Specific to gcc __attribute__ tells gcc the function takes vararg parameters that are printf format specifiers.
  int VPP(const char *iFmt, ... ) __attribute__((format(printf, 2, 3))) {
    int rc = 0;
    if (iFmt) {
      #pragma omp critical (Verb)
      {
        fflush(stdout); fflush(stderr);
        if (*VerbPrefix2(& mV)) printf("%s: ", VerbPrefix2(& mV));
        va_list list;
        va_start(list, iFmt);
        rc = vprintf(iFmt, list);
        va_end(list);
        fflush(stdout);
      }
    }
    return rc;
  }

  // Like the above but onlocked, it's up to the user to lock with
  //#pragma omp critical (Verb)
  int WP(const char *iFmt, ... ) __attribute__((format(printf, 2, 3))) {
    int rc = 0;
    if (iFmt) {
      fflush(stdout); fflush(stderr);
      if (*VerbPrefix2(& mV)) printf("%s: ", VerbPrefix2(& mV));
      va_list list;
      va_start(list, iFmt);
      rc = vprintf(iFmt, list);
      va_end(list);
      fflush(stdout);
    }
    return rc;
  }
};

// Ewma track mean and variance for M values
template <int M, int N> struct EwmaMuVar {

  static const int M_M = M;
  static const int M_N = N;

  double (*mEwmaVal_MxN)[M][N] = nullptr, (*mEwmaVal2_MxN)[M][N] = nullptr;
  double mDecay = 0.9, mDecay1 = 1 - mDecay, mThr1 = 100, mThr2 = 100;
  int mNum = 0;

  void Reset(double iVal, double iVal2) {
    #pragma omp critical
    {
      mNum = 0;
      for (int iM = 0; iM < M; ++iM) {
        for (int iN = 0; iN < N; ++iN) {
          (*mEwmaVal_MxN) [iM][iN] = iVal;
          (*mEwmaVal2_MxN)[iM][iN] = iVal2;
        }
      }
    }
  }
  void Reset(void) { Reset(0, 0); }

  EwmaMuVar(double iDecay, double iThr1, double iThr2, double iVal, double iVal2) : mDecay(iDecay), mDecay1(1 - iDecay), mThr1(iThr1), mThr2(iThr2) {
    XASSERT(0 < mDecay  && mDecay  < 1);
    XASSERT(0 < mDecay1 && mDecay1 < 1);
    XASSERT(std::fabs(mDecay + mDecay1 - 1) < 1e-20);
    XASSERT(mThr1 > 0);
    XASSERT(mThr2 > 0);
    XASSERT(M > 0);
    XASSERT(N > 0);
    mEwmaVal_MxN  = (double (*)[M][N])XMALLOC(double, M * N);
    mEwmaVal2_MxN = (double (*)[M][N])XMALLOC(double, M * N);
    Reset(iVal, iVal2);
  }

  ~EwmaMuVar() {
    XFREE(mEwmaVal_MxN);
    XFREE(mEwmaVal2_MxN);
  }

  bool Accumulate(int iM, int iN, double iVal) {
    if(! std::isnan(iVal)) {
      #pragma omp critical
      {
        (*mEwmaVal_MxN) [iM][iN] = mDecay1 * iVal        + mDecay * (*mEwmaVal_MxN) [iM][iN];
        (*mEwmaVal2_MxN)[iM][iN] = mDecay1 * iVal * iVal + mDecay * (*mEwmaVal2_MxN)[iM][iN];
        ++mNum;
      }
      return true;
    }
    return false;
  }

  // Return indicator how distant is a value from the rolling mean/var
  //  0 = within the thresholds range
  // -1 = below ratio 1/thr1 on the 1/side relative to mu only
  //  1 = above ratio thr1/1 on the side/1 relative to mu only
  // -2 = below thr2 on the -ve side relative to mu/std
  //  2 = above thr2 on the +ve side relative to mu/std
  int Distance(int iM, int iN, double iVal, double *oDist, double *oMu, double *oStd) {
    double var = (*mEwmaVal2_MxN)[iM][iN] - (*mEwmaVal_MxN)[iM][iN] * (*mEwmaVal_MxN)[iM][iN];
    // This should always be the case, but check jic and have a fallback
    if (var > 0) {
      double st = std::sqrt(var);
      double d = (iVal - (*mEwmaVal_MxN)[iM][iN]) / st;
      if (oDist) { *oDist = d; }
      if (oMu) { *oMu = (*mEwmaVal_MxN)[iM][iN]; }
      if (oStd) { *oStd = st; }
      if (d < -mThr2) { return -2; }
      if (mThr2 < d) { return 2; }
    } else {
      double d = iVal / (*mEwmaVal_MxN)[iM][iN];
      if (oDist) { *oDist = d; }
      if (oMu) { *oMu = (*mEwmaVal_MxN)[iM][iN]; }
      if (mThr1 < abs(d)) { return 1; }
      if (abs(d) < 1/mThr1) { return -1; }
    }
    return 0;
  }

};

// Mini tallies tracker keeper
struct PosNeg {
  int mPosNum = 0, mNegNum = 0, mNum = 0, mNanNum = 0, mZeroNum = 0;
  double mPosSum = 0, mNegSum = 0;
  void Add(double iVal) {
    ++mNum;
    if (std::isnan(iVal)) {
      ++mNanNum;
    } else if (iVal > 0) {
      ++mPosNum;
      mPosSum += iVal;
    } else if (iVal < 0) {
      ++mNegNum;
      mNegSum += iVal;
    } else {
      ++mZeroNum;
    }
  }
};

// Beautiful Branchless Binary Search by Malte Skarupke
// https://probablydance.com/2023/04/27/beautiful-branchless-binary-search/
// https://github.com/skarupke/branchless_binary_search
// Commented out as it fails to compile needs C++20 for std::countl_zero().
#if 0
inline size_t bit_floor(size_t i) {
  constexpr int num_bits = sizeof(i) * 8;
  return size_t(1) << (num_bits - std::countl_zero(i) - 1);
}
inline size_t bit_ceil(size_t i) {
  constexpr int num_bits = sizeof(i) * 8;
  return size_t(1) << (num_bits - std::countl_zero(i - 1));
}
template<typename It, typename T, typename Cmp>
It branchless_lower_bound2(It begin, It end, const T & value, Cmp && compare) {
  size_t length = end - begin;
  if (length == 0) return end;
  size_t step = bit_floor(length);
  if (step != length && compare(begin[step], value)) {
    length -= step + 1;
    if (length == 0) { return end; }
    step = bit_ceil(length);
    begin = end - step;
  }
  for (step /= 2; step != 0; step /= 2) {
      if (compare(begin[step], value)) { begin += step; }
  }
  return begin + compare(*begin, value);
}
template<typename It, typename T>
It branchless_lower_bound2(It begin, It end, const T & value) {
	return branchless_lower_bound2(begin, end, value, std::less<>{});
}
#endif

// https://stackoverflow.com/questions/14454592/how-do-i-perform-an-almost-branch-less-binary-search-on-arbitrary-sorted-data
template<class FwdIt, class T, class P>
FwdIt branchless_lower_bound3(FwdIt begin, FwdIt end, T const & val, P pred) {
	while (begin != end) {
		FwdIt middle(begin);
		std::advance(middle, std::distance(begin, end) >> 1);
		FwdIt middle_plus_one(middle);
		++middle_plus_one;
		bool const b = pred(*middle, val);
		begin = b ? middle_plus_one : begin;
		end = b ? end : middle;
	}
	return begin;
}
template<class FwdIt, class T, class P>
FwdIt branchless_lower_bound3(FwdIt begin, FwdIt end, const T & val) {
	return branchless_lower_bound3(begin, end, val, std::less<>{});
}

#endif // #ifndef XCOMMON_HPP

