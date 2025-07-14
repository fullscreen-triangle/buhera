# VPOS Semantic Processing Framework Makefile
#
# Copyright (c) 2024 VPOS Development Team
# Licensed under MIT License

obj-m += semantic-processor.o

# Module source files
semantic-processor-objs := semantic-processor.o

# Kernel build directory
KERNEL_DIR ?= /lib/modules/$(shell uname -r)/build

# Build flags
ccflags-y := -I$(src)/../../../include
ccflags-y += -I$(src)/../quantum
ccflags-y += -I$(src)/../bmd
ccflags-y += -I$(src)/../temporal
ccflags-y += -DVPOS_SEMANTIC_PROCESSING
ccflags-y += -DCONFIG_VPOS_SEMANTIC_NEURAL_NETWORK
ccflags-y += -DCONFIG_VPOS_SEMANTIC_CONSCIOUSNESS
ccflags-y += -DCONFIG_VPOS_SEMANTIC_BMD_INTEGRATION
ccflags-y += -O2
ccflags-y += -Wall
ccflags-y += -Wextra
ccflags-y += -std=gnu99

# Debug flags (uncomment for debugging)
# ccflags-y += -DDEBUG
# ccflags-y += -g

# Architecture-specific optimizations
ifeq ($(ARCH),x86_64)
    ccflags-y += -march=native
    ccflags-y += -mtune=native
    ccflags-y += -msse4.2
    ccflags-y += -mavx2
endif

# Semantic processing specific defines
ccflags-y += -DSEMANTIC_INPUT_NEURONS=1024
ccflags-y += -DSEMANTIC_HIDDEN_NEURONS=512
ccflags-y += -DSEMANTIC_HIDDEN_LAYERS=3
ccflags-y += -DSEMANTIC_OUTPUT_NEURONS=256
ccflags-y += -DSEMANTIC_CONTEXT_HASH_SIZE=1024
ccflags-y += -DSEMANTIC_CACHE_HASH_SIZE=2048
ccflags-y += -DSEMANTIC_MAX_OPERATIONS=10000

# Performance optimization flags
ccflags-y += -DCONFIG_VPOS_SEMANTIC_OPTIMIZE_NEURAL_NETWORK
ccflags-y += -DCONFIG_VPOS_SEMANTIC_OPTIMIZE_CONTEXT_DB
ccflags-y += -DCONFIG_VPOS_SEMANTIC_OPTIMIZE_MEANING_ENGINE
ccflags-y += -DCONFIG_VPOS_SEMANTIC_OPTIMIZE_CACHE

# Security flags
ccflags-y += -DCONFIG_VPOS_SEMANTIC_SECURITY_CHECKS
ccflags-y += -DCONFIG_VPOS_SEMANTIC_MEMORY_PROTECTION
ccflags-y += -DCONFIG_VPOS_SEMANTIC_INPUT_VALIDATION

# Build targets
all: modules

modules:
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) modules

modules_install:
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) modules_install

clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) clean
	rm -f *.o *.ko *.mod.c *.mod.o *.symvers *.order
	rm -rf .tmp_versions

# Install target
install: modules_install
	depmod -a
	modprobe semantic-processor

# Uninstall target
uninstall:
	rmmod semantic-processor || true
	rm -f /lib/modules/$(shell uname -r)/extra/semantic-processor.ko
	depmod -a

# Test target
test:
	@echo "Testing semantic processing framework..."
	@if lsmod | grep -q semantic_processor; then \
		echo "Semantic processor module loaded successfully"; \
	else \
		echo "Semantic processor module not loaded"; \
		exit 1; \
	fi
	@dmesg | tail -20 | grep -i "semantic"

# Debug target
debug:
	$(MAKE) ccflags-y+="-DDEBUG -g" modules

# Help target
help:
	@echo "VPOS Semantic Processing Framework Build System"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all modules (default)"
	@echo "  modules      - Build kernel modules"
	@echo "  clean        - Clean build artifacts"
	@echo "  install      - Install modules and load"
	@echo "  uninstall    - Remove modules"
	@echo "  test         - Test module loading"
	@echo "  debug        - Build with debug symbols"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Build configuration:"
	@echo "  Neural Network: $(SEMANTIC_INPUT_NEURONS) -> $(SEMANTIC_HIDDEN_NEURONS) (x$(SEMANTIC_HIDDEN_LAYERS)) -> $(SEMANTIC_OUTPUT_NEURONS)"
	@echo "  Context Hash Size: $(SEMANTIC_CONTEXT_HASH_SIZE)"
	@echo "  Cache Hash Size: $(SEMANTIC_CACHE_HASH_SIZE)"
	@echo "  Max Operations: $(SEMANTIC_MAX_OPERATIONS)"
	@echo ""
	@echo "Features enabled:"
	@echo "  - Neural Network Processing"
	@echo "  - Context Database"
	@echo "  - Meaning Preservation Engine"
	@echo "  - BMD Integration"
	@echo "  - Consciousness Interface"
	@echo "  - Semantic Cache System"
	@echo "  - Quality Assessment"
	@echo "  - Performance Optimization"
	@echo "  - Security Checks"

# Module information
info:
	@echo "VPOS Semantic Processing Framework"
	@echo "================================="
	@echo "Version: 1.0.0"
	@echo "License: MIT"
	@echo "Authors: VPOS Development Team"
	@echo ""
	@echo "Module: semantic-processor.ko"
	@echo "Description: Revolutionary meaning-preserving computational transformations"
	@echo "Dependencies: quantum-coherence, bmd-catalyst, temporal-masunda"
	@echo ""
	@echo "Capabilities:"
	@echo "  - Multi-layer neural network (1024->512x3->256)"
	@echo "  - Context-aware semantic processing"
	@echo "  - Meaning preservation algorithms"
	@echo "  - BMD information catalysis integration"
	@echo "  - Consciousness-aware transformations"
	@echo "  - High-performance semantic cache"
	@echo "  - Quality assessment metrics"
	@echo "  - Real-time pattern recognition"

.PHONY: all modules modules_install clean install uninstall test debug help info 