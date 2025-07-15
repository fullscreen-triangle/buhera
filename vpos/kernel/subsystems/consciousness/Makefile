# VPOS Consciousness Integration Subsystem Makefile
# Revolutionary consciousness theoretical framework integrated into operational system
# Build configuration for consciousness-aware processing capabilities

# Compiler Configuration
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O3 -g -fPIC
LDFLAGS = -shared -lm -lpthread -lrt

# Architecture-specific optimizations
ifeq ($(shell uname -m), x86_64)
    CFLAGS += -march=native -mtune=native -msse4.2 -mavx2
endif

# Directories
SRCDIR = .
OBJDIR = obj
BINDIR = bin
INCDIR = .
LIBDIR = lib

# Include paths
INCLUDES = -I$(INCDIR) \
           -I../../core/bmd \
           -I../../core/temporal \
           -I../../core/quantum \
           -I../../core/neural \
           -I../../core/semantic \
           -I../../core/fuzzy \
           -I../../core/vpos \
           -I../../../include

# Source files
SOURCES = consciousness-integration.c

# Object files
OBJECTS = $(SOURCES:%.c=$(OBJDIR)/%.o)

# Target library
TARGET = $(LIBDIR)/libvpos-consciousness-integration.so

# Test executable
TEST_TARGET = $(BINDIR)/consciousness-integration-test

# Dependencies
DEPS = consciousness-integration.h

# Performance test executable
PERF_TARGET = $(BINDIR)/consciousness-integration-performance

# Integration test executable
INTEGRATION_TARGET = $(BINDIR)/consciousness-integration-integration

# Benchmark executable
BENCHMARK_TARGET = $(BINDIR)/consciousness-integration-benchmark

# Default target
all: directories $(TARGET) $(TEST_TARGET) $(PERF_TARGET) $(INTEGRATION_TARGET) $(BENCHMARK_TARGET)

# Create directories
directories:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(BINDIR)
	@mkdir -p $(LIBDIR)

# Compile object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Link target library
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

# Test executable
$(TEST_TARGET): $(TARGET) test/consciousness-integration-test.c
	$(CC) $(CFLAGS) $(INCLUDES) test/consciousness-integration-test.c -L$(LIBDIR) -lvpos-consciousness-integration $(LDFLAGS) -o $@

# Performance test executable
$(PERF_TARGET): $(TARGET) test/consciousness-integration-performance.c
	$(CC) $(CFLAGS) $(INCLUDES) test/consciousness-integration-performance.c -L$(LIBDIR) -lvpos-consciousness-integration $(LDFLAGS) -o $@

# Integration test executable
$(INTEGRATION_TARGET): $(TARGET) test/consciousness-integration-integration.c
	$(CC) $(CFLAGS) $(INCLUDES) test/consciousness-integration-integration.c -L$(LIBDIR) -lvpos-consciousness-integration $(LDFLAGS) -o $@

# Benchmark executable
$(BENCHMARK_TARGET): $(TARGET) test/consciousness-integration-benchmark.c
	$(CC) $(CFLAGS) $(INCLUDES) test/consciousness-integration-benchmark.c -L$(LIBDIR) -lvpos-consciousness-integration $(LDFLAGS) -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR) $(LIBDIR)
	rm -f *.o *.so *.a
	rm -f core.*
	rm -f test/*.o
	rm -f test/core.*

# Install target
install: $(TARGET)
	cp $(TARGET) /usr/local/lib/
	cp consciousness-integration.h /usr/local/include/vpos/
	ldconfig

# Uninstall target
uninstall:
	rm -f /usr/local/lib/libvpos-consciousness-integration.so
	rm -f /usr/local/include/vpos/consciousness-integration.h
	ldconfig

# Test targets
test: $(TEST_TARGET)
	./$(TEST_TARGET)

performance: $(PERF_TARGET)
	./$(PERF_TARGET)

integration: $(INTEGRATION_TARGET)
	./$(INTEGRATION_TARGET)

benchmark: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

# Development targets
debug: CFLAGS += -DDEBUG -g3 -O0
debug: clean all

profile: CFLAGS += -pg -fprofile-arcs -ftest-coverage
profile: LDFLAGS += -pg -lgcov
profile: clean all

# Coverage analysis
coverage: profile
	./$(TEST_TARGET)
	gcov $(SOURCES)
	lcov -c -d . -o coverage.info
	genhtml coverage.info -o coverage_report

# Memory leak detection
valgrind: $(TEST_TARGET)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(TEST_TARGET)

# Static analysis
analyze:
	clang-tidy $(SOURCES) -- $(CFLAGS) $(INCLUDES)
	cppcheck --enable=all --std=c99 $(SOURCES)

# Documentation generation
docs:
	doxygen Doxyfile

# Code formatting
format:
	clang-format -i $(SOURCES) $(DEPS)

# Continuous integration
ci: clean all test performance benchmark analyze

# Dependencies tracking
depend: $(SOURCES)
	$(CC) -MM $(CFLAGS) $(INCLUDES) $^ > .depend

# Include dependencies if they exist
-include .depend

# Help target
help:
	@echo "VPOS Consciousness Integration Subsystem Build System"
	@echo "Available targets:"
	@echo "  all         - Build library and test executables"
	@echo "  clean       - Remove all build artifacts"
	@echo "  install     - Install library and headers"
	@echo "  uninstall   - Remove installed files"
	@echo "  test        - Run basic functionality tests"
	@echo "  performance - Run performance tests"
	@echo "  integration - Run integration tests"
	@echo "  benchmark   - Run comprehensive benchmarks"
	@echo "  debug       - Build with debug symbols"
	@echo "  profile     - Build with profiling support"
	@echo "  coverage    - Generate code coverage report"
	@echo "  valgrind    - Run memory leak detection"
	@echo "  analyze     - Run static code analysis"
	@echo "  docs        - Generate documentation"
	@echo "  format      - Format source code"
	@echo "  ci          - Continuous integration build"
	@echo "  help        - Show this help message"

# Build information
info:
	@echo "VPOS Consciousness Integration Subsystem"
	@echo "Version: 1.0.0"
	@echo "Compiler: $(CC)"
	@echo "Flags: $(CFLAGS)"
	@echo "Includes: $(INCLUDES)"
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Target: $(TARGET)"

# Phony targets
.PHONY: all clean install uninstall test performance integration benchmark debug profile coverage valgrind analyze docs format ci depend help info directories 