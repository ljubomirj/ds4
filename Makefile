CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -Wall -Wextra -fobjc-arc
LDLIBS ?= -lm -framework Foundation -framework Metal

.PHONY: all clean test

all: test_l26f test_l26f_multilayer

l26f_gguf.o: l26f_gguf.c l26f.h
	$(CC) $(CFLAGS) -c -o $@ l26f_gguf.c

l26f_metal.o: l26f_metal.m l26f_metal.h
	$(CC) $(OBJCFLAGS) -c -o $@ l26f_metal.m

l26f.o: l26f.c l26f.h l26f_metal.h
	$(CC) $(CFLAGS) -c -o $@ l26f.c

test_l26f: test_l26f.o l26f.o l26f_gguf.o l26f_metal.o
	$(CC) $(CFLAGS) -o $@ test_l26f.o l26f.o l26f_gguf.o l26f_metal.o $(LDLIBS)

test_l26f.o: test_l26f.c l26f.h
	$(CC) $(CFLAGS) -c -o $@ test_l26f.c

test_l26f_multilayer: test_l26f_multilayer.o l26f.o l26f_gguf.o l26f_metal.o l26f_mla_cpu.o l26f_tokenizer.o
	$(CC) $(CFLAGS) -o $@ test_l26f_multilayer.o l26f.o l26f_gguf.o l26f_metal.o l26f_mla_cpu.o l26f_tokenizer.o $(LDLIBS)

test_l26f_multilayer.o: test_l26f_multilayer.c l26f.h l26f_metal.h ds4_metal.h l26f_tokenizer.h
	$(CC) $(CFLAGS) -c -o $@ test_l26f_multilayer.c

l26f_mla_cpu.o: l26f_mla_cpu.c l26f.h
	$(CC) $(CFLAGS) -c -o $@ l26f_mla_cpu.c

l26f_tokenizer.o: l26f_tokenizer.c l26f_tokenizer.h
	$(CC) $(CFLAGS) -c -o $@ l26f_tokenizer.c

test: test_l26f
	./test_l26f

clean:
	rm -f *.o test_l26f test_l26f_multilayer
