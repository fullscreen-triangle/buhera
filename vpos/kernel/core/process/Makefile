# VPOS Virtual Processor Manager Makefile
#
# Copyright (c) 2024 VPOS Development Team
# Licensed under MIT License

obj-m += virtual-processor-manager.o

# Module source files
virtual-processor-manager-objs := virtual-processor-manager.o

# Kernel build directory
KERNEL_DIR ?= /lib/modules/$(shell uname -r)/build

# Build flags
ccflags-y := -I$(src)/../../../include
ccflags-y += -I$(src)/../scheduler
ccflags-y += -I$(src)/../quantum
ccflags-y += -I$(src)/../memory
ccflags-y += -I$(src)/../temporal
ccflags-y += -I$(src)/../bmd
ccflags-y += -I$(src)/../semantic
ccflags-y += -I$(src)/../../subsystems/neural-transfer
ccflags-y += -I$(src)/../../subsystems/molecular-foundry
ccflags-y += -DVPOS_VIRTUAL_PROCESSOR_MANAGER
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_MANAGEMENT
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_SCHEDULING
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_MIGRATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_LOAD_BALANCING
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_PERFORMANCE
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_COMMUNICATION
ccflags-y += -O2
ccflags-y += -Wall
ccflags-y += -Wextra
ccflags-y += -std=gnu99

# Virtual processor manager specific defines
ccflags-y += -DVIRTUAL_PROCESSOR_POOL_COUNT=8
ccflags-y += -DVIRTUAL_PROCESSOR_POOL_SIZE=256
ccflags-y += -DVIRTUAL_PROCESSOR_MAX_COUNT=2048
ccflags-y += -DVIRTUAL_PROCESSOR_PRIORITY_LEVELS=16
ccflags-y += -DVIRTUAL_PROCESSOR_STATES=8
ccflags-y += -DVIRTUAL_PROCESSOR_TYPES=8
ccflags-y += -DVIRTUAL_PROCESSOR_TIMESLICE_NS=2000000
ccflags-y += -DVIRTUAL_PROCESSOR_PREEMPTION_THRESHOLD=1000000
ccflags-y += -DVIRTUAL_PROCESSOR_MIGRATION_THRESHOLD=10000000
ccflags-y += -DVIRTUAL_PROCESSOR_LOAD_BALANCE_INTERVAL=50000000

# Performance optimization flags
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_OPTIMIZE_CREATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_OPTIMIZE_SCHEDULING
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_OPTIMIZE_MIGRATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_OPTIMIZE_COMMUNICATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_OPTIMIZE_PERFORMANCE

# Security flags
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_SECURITY_CHECKS
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_ISOLATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_PROTECTION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_VALIDATION

# Advanced features
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_QUANTUM_INTEGRATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_NEURAL_INTEGRATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_FUZZY_SCHEDULING
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_TEMPORAL_COORDINATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_BMD_INTEGRATION
ccflags-y += -DCONFIG_VPOS_VIRTUAL_PROCESSOR_SEMANTIC_INTEGRATION

# Build targets
all: modules

modules:
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) modules

clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) clean

install: modules
	$(MAKE) -C $(KERNEL_DIR) M=$(shell pwd) modules_install
	depmod -a
	modprobe virtual-processor-manager

uninstall:
	rmmod virtual-processor-manager || true
	rm -f /lib/modules/$(shell uname -r)/extra/virtual-processor-manager.ko
	depmod -a

test:
	@echo "Testing virtual processor manager..."
	@if lsmod | grep -q virtual_processor_manager; then \
		echo "Virtual processor manager module loaded successfully"; \
	else \
		echo "Virtual processor manager module not loaded"; \
		exit 1; \
	fi

help:
	@echo "VPOS Virtual Processor Manager Build System"
	@echo "==========================================="
	@echo "Available targets:"
	@echo "  all      - Build all modules (default)"
	@echo "  modules  - Build kernel modules"
	@echo "  clean    - Clean build artifacts"
	@echo "  install  - Install modules and load"
	@echo "  uninstall - Remove modules"
	@echo "  test     - Test module loading"
	@echo "  help     - Show this help"

info:
	@echo "VPOS Virtual Processor Manager"
	@echo "============================="
	@echo "Version: 1.0.0"
	@echo "Description: Advanced virtual processor management and coordination"
	@echo "Features:"
	@echo "  - Virtual processor pools (8 pools, 256 processors each)"
	@echo "  - Dynamic virtual processor lifecycle management"
	@echo "  - Fuzzy quantum scheduler integration"
	@echo "  - Virtual processor migration and load balancing"
	@echo "  - Performance monitoring and optimization"
	@echo "  - Inter-processor communication"
	@echo "  - Resource management and allocation"
	@echo "  - Priority-based scheduling"
	@echo "  - CPU affinity management"
	@echo "  - Context switching optimization"
	@echo "  - Quantum integration support"
	@echo "  - Neural processing integration"
	@echo "  - Temporal coordination"
	@echo "  - BMD and semantic integration"
	@echo "  - Security and isolation features"
	@echo "  - Real-time performance metrics"

.PHONY: all modules clean install uninstall test help info 