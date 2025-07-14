# VPOS Neural Pattern Transfer System Makefile
#
# Copyright (c) 2024 VPOS Development Team
# Licensed under MIT License

obj-m += neural-pattern-transfer.o

# Module source files
neural-pattern-transfer-objs := neural-pattern-transfer.o

# Kernel build directory
KERNEL_DIR ?= /lib/modules/$(shell uname -r)/build

# Build flags
ccflags-y := -I$(src)/../../../include
ccflags-y += -I$(src)/../../core/bmd
ccflags-y += -I$(src)/../../core/semantic
ccflags-y += -I$(src)/../../core/quantum
ccflags-y += -I$(src)/../../core/temporal
ccflags-y += -DVPOS_NEURAL_PATTERN_TRANSFER
ccflags-y += -DCONFIG_VPOS_NEURAL_EXTRACTION
ccflags-y += -DCONFIG_VPOS_MEMORY_INJECTION
ccflags-y += -DCONFIG_VPOS_CONSCIOUSNESS_TRANSFER
ccflags-y += -DCONFIG_VPOS_BMD_NEURAL_EXTRACTION
ccflags-y += -DCONFIG_VPOS_NEURAL_SYNTHESIS
ccflags-y += -O2
ccflags-y += -Wall
ccflags-y += -Wextra
ccflags-y += -std=gnu99

# Neural pattern transfer specific defines
ccflags-y += -DNEURAL_PATTERN_DATABASE_SIZE=4096
ccflags-y += -DMEMORY_INJECTION_UNIT_COUNT=32
ccflags-y += -DPATTERN_SYNTHESIS_REACTOR_COUNT=16
ccflags-y += -DNEURAL_SUBSTRATE_ARRAY_SIZE=64
ccflags-y += -DCONSCIOUSNESS_INTERFACE_COUNT=8
ccflags-y += -DBMD_EXTRACTION_CHANNEL_COUNT=16

# Performance optimization flags
ccflags-y += -DCONFIG_VPOS_NEURAL_OPTIMIZE_EXTRACTION
ccflags-y += -DCONFIG_VPOS_NEURAL_OPTIMIZE_INJECTION
ccflags-y += -DCONFIG_VPOS_NEURAL_OPTIMIZE_ANALYSIS
ccflags-y += -DCONFIG_VPOS_NEURAL_OPTIMIZE_SYNTHESIS
ccflags-y += -DCONFIG_VPOS_NEURAL_OPTIMIZE_BMD
ccflags-y += -DCONFIG_VPOS_NEURAL_OPTIMIZE_CONSCIOUSNESS

# Security flags
ccflags-y += -DCONFIG_VPOS_NEURAL_SECURITY_CHECKS
ccflags-y += -DCONFIG_VPOS_NEURAL_MEMORY_PROTECTION
ccflags-y += -DCONFIG_VPOS_NEURAL_INPUT_VALIDATION
ccflags-y += -DCONFIG_VPOS_NEURAL_CONSCIOUSNESS_PROTECTION

# Build targets
all: modules

modules:
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) modules

clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) clean

install: modules
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) modules_install
	depmod -a
	modprobe neural-pattern-transfer

uninstall:
	rmmod neural-pattern-transfer || true
	rm -f /lib/modules/$(shell uname -r)/extra/neural-pattern-transfer.ko
	depmod -a

test:
	@echo "Testing neural pattern transfer system..."
	@if lsmod | grep -q neural_pattern_transfer; then \
		echo "Neural pattern transfer module loaded successfully"; \
	else \
		echo "Neural pattern transfer module not loaded"; \
		exit 1; \
	fi

help:
	@echo "VPOS Neural Pattern Transfer System Build System"
	@echo "=============================================="
	@echo "Available targets:"
	@echo "  all      - Build all modules (default)"
	@echo "  modules  - Build kernel modules"
	@echo "  clean    - Clean build artifacts"
	@echo "  install  - Install modules and load"
	@echo "  uninstall - Remove modules"
	@echo "  test     - Test module loading"
	@echo "  help     - Show this help"

info:
	@echo "VPOS Neural Pattern Transfer System"
	@echo "=================================="
	@echo "Version: 1.0.0"
	@echo "Description: Revolutionary neural pattern extraction and memory injection"
	@echo "Features:"
	@echo "  - BMD-mediated neural pattern processing"
	@echo "  - Consciousness transfer protocols"
	@echo "  - Memory injection units (32)"
	@echo "  - Pattern synthesis reactors (16)"
	@echo "  - Neural pattern database (4096 entries)"
	@echo "  - Multi-type memory support"
	@echo "  - Consciousness extraction/injection"
	@echo "  - Neural plasticity monitoring"
	@echo "  - Synaptic pattern recognition"

.PHONY: all modules clean install uninstall test help info 