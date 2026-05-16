CC ?= cc
CFLAGS_COMMON := -Wall -Wextra -std=c99
OBJCFLAGS_COMMON := -Wall -Wextra -fobjc-arc
LDLIBS ?= -lm -framework Foundation -framework Metal

BUILDDIR ?= build_release
OPT_CFLAGS ?= -O3
EXTRA_CFLAGS ?=

CFLAGS := $(CFLAGS_COMMON) $(OPT_CFLAGS) $(EXTRA_CFLAGS)
OBJCFLAGS := $(OBJCFLAGS_COMMON) $(OPT_CFLAGS) $(EXTRA_CFLAGS)

PUBLIC_DIR := public
WEBUI_ASSETS := index.html bundle.js bundle.css loading.html
WEBUI_HEADERS := $(foreach a,$(WEBUI_ASSETS),$(BUILDDIR)/$(a).h)

.PHONY: all release debug clean test webui

all: release

release:
	$(MAKE) test_l26f_multilayer l26f_server BUILDDIR=build_release OPT_CFLAGS=-O3

debug:
	$(MAKE) test_l26f_multilayer l26f_server BUILDDIR=build_debug OPT_CFLAGS="-O0 -g -DL26F_DEBUG"

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

webui: $(WEBUI_HEADERS)

$(BUILDDIR)/%.h: $(PUBLIC_DIR)/% | $(BUILDDIR)
	xxd -i $< | sed 's/unsigned char public_/unsigned char l26f_webui_/;s/unsigned int public_/unsigned int l26f_webui_/' > $@

COMMON_OBJS = $(BUILDDIR)/l26f.o $(BUILDDIR)/l26f_gguf.o $(BUILDDIR)/l26f_metal.o \
              $(BUILDDIR)/l26f_mla_gpu.o $(BUILDDIR)/l26f_mla_cpu.o $(BUILDDIR)/l26f_tokenizer.o \
              $(BUILDDIR)/xcommon.o

$(BUILDDIR)/l26f_gguf.o: l26f_gguf.c l26f.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ l26f_gguf.c

$(BUILDDIR)/l26f_metal.o: l26f_metal.m l26f_metal.h | $(BUILDDIR)
	$(CC) $(OBJCFLAGS) -c -o $@ l26f_metal.m

$(BUILDDIR)/l26f.o: l26f.c l26f.h l26f_metal.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ l26f.c

$(BUILDDIR)/l26f_mla_gpu.o: l26f_mla_gpu.c l26f.h l26f_metal.h ds4_metal.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ l26f_mla_gpu.c

$(BUILDDIR)/l26f_mla_cpu.o: l26f_mla_cpu.c l26f.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ l26f_mla_cpu.c

$(BUILDDIR)/l26f_tokenizer.o: l26f_tokenizer.c l26f_tokenizer.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ l26f_tokenizer.c

$(BUILDDIR)/xcommon.o: xcommon.c xcommon.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ xcommon.c

$(BUILDDIR)/test_l26f_multilayer.o: test_l26f_multilayer.c l26f.h l26f_metal.h ds4_metal.h l26f_tokenizer.h xcommon.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ test_l26f_multilayer.c

test_l26f_multilayer: $(BUILDDIR)/test_l26f_multilayer.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $(BUILDDIR)/$@ $^ $(LDLIBS)

$(BUILDDIR)/l26f_server.o: l26f_server.c test_l26f_multilayer.c l26f.h l26f_metal.h ds4_metal.h l26f_tokenizer.h xcommon.h $(WEBUI_HEADERS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ l26f_server.c

l26f_server: $(BUILDDIR)/l26f_server.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $(BUILDDIR)/$@ $^ $(LDLIBS)

$(BUILDDIR)/test_l26f.o: test_l26f.c l26f.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ test_l26f.c

test_l26f: $(BUILDDIR)/test_l26f.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $(BUILDDIR)/$@ $^ $(LDLIBS)

$(BUILDDIR)/test_l26f_tokenize.o: test_l26f_tokenize.c l26f.h l26f_tokenizer.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ test_l26f_tokenize.c

test_l26f_tokenize: $(BUILDDIR)/test_l26f_tokenize.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $(BUILDDIR)/$@ $^ $(LDLIBS)

test: test_l26f
	./$(BUILDDIR)/test_l26f

clean:
	rm -rf build_debug build_release *.o test_l26f test_l26f_multilayer
