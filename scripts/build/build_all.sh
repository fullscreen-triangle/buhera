#!/bin/bash

# Buhera VPOS Gas Oscillation Server Farm Build Script
# This script builds all components of the server farm system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/target"
VPOS_DIR="${PROJECT_ROOT}/vpos"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Create directories if they don't exist
mkdir -p "${LOGS_DIR}"
mkdir -p "${BUILD_DIR}"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking build dependencies..."
    
    # Check for Rust and Cargo
    if ! command_exists rustc; then
        print_error "Rust is not installed. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    if ! command_exists cargo; then
        print_error "Cargo is not installed. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    # Check for build tools
    if ! command_exists make; then
        print_error "Make is not installed. Please install build-essential or equivalent"
        exit 1
    fi
    
    if ! command_exists gcc; then
        print_error "GCC is not installed. Please install build-essential or equivalent"
        exit 1
    fi
    
    # Check for pnpm (user preference)
    if ! command_exists pnpm; then
        print_warning "pnpm is not installed. Some components may use npm instead"
    fi
    
    print_success "All dependencies are available"
}

# Function to build kernel modules
build_kernel_modules() {
    print_status "Building VPOS kernel modules..."
    
    # Build gas oscillation kernel modules
    if [ -d "${VPOS_DIR}/kernel/core/gas_oscillation" ]; then
        print_status "Building gas oscillation kernel modules..."
        cd "${VPOS_DIR}/kernel/core/gas_oscillation"
        
        if [ -f "Makefile" ]; then
            make clean 2>/dev/null || true
            make 2>&1 | tee "${LOGS_DIR}/gas_oscillation_kernel.log"
            
            if [ $? -eq 0 ]; then
                print_success "Gas oscillation kernel modules built successfully"
            else
                print_error "Failed to build gas oscillation kernel modules"
                return 1
            fi
        else
            print_warning "No Makefile found in gas_oscillation directory"
        fi
    else
        print_warning "Gas oscillation kernel directory not found"
    fi
    
    # Build gas oscillation drivers
    if [ -d "${VPOS_DIR}/kernel/drivers/gas_oscillation" ]; then
        print_status "Building gas oscillation drivers..."
        cd "${VPOS_DIR}/kernel/drivers/gas_oscillation"
        
        if [ -f "Makefile" ]; then
            make clean 2>/dev/null || true
            make 2>&1 | tee "${LOGS_DIR}/gas_oscillation_drivers.log"
            
            if [ $? -eq 0 ]; then
                print_success "Gas oscillation drivers built successfully"
            else
                print_error "Failed to build gas oscillation drivers"
                return 1
            fi
        else
            print_warning "No Makefile found in gas_oscillation drivers directory"
        fi
    else
        print_warning "Gas oscillation drivers directory not found"
    fi
    
    # Build other kernel modules
    print_status "Building other kernel modules..."
    
    # Build quantum kernel modules
    if [ -d "${VPOS_DIR}/kernel/core/quantum" ]; then
        cd "${VPOS_DIR}/kernel/core/quantum"
        if [ -f "Makefile" ]; then
            make clean 2>/dev/null || true
            make 2>&1 | tee "${LOGS_DIR}/quantum_kernel.log"
        fi
    fi
    
    # Build scheduler modules
    if [ -d "${VPOS_DIR}/kernel/core/scheduler" ]; then
        cd "${VPOS_DIR}/kernel/core/scheduler"
        if [ -f "Makefile" ]; then
            make clean 2>/dev/null || true
            make 2>&1 | tee "${LOGS_DIR}/scheduler_kernel.log"
        fi
    fi
    
    print_success "Kernel modules build completed"
}

# Function to build Rust components
build_rust_components() {
    print_status "Building Rust components..."
    
    cd "${PROJECT_ROOT}"
    
    # Check if Cargo.toml exists
    if [ ! -f "Cargo.toml" ]; then
        print_error "Cargo.toml not found in project root"
        return 1
    fi
    
    # Build with all features
    print_status "Building with all server farm features..."
    cargo build --release --all-features 2>&1 | tee "${LOGS_DIR}/rust_build.log"
    
    if [ $? -eq 0 ]; then
        print_success "Rust components built successfully"
    else
        print_error "Failed to build Rust components"
        return 1
    fi
    
    # Build specific binaries
    print_status "Building server farm binaries..."
    
    # Build server farm binary
    cargo build --release --bin buhera-server-farm 2>&1 | tee "${LOGS_DIR}/server_farm_binary.log"
    if [ $? -eq 0 ]; then
        print_success "Server farm binary built successfully"
    else
        print_error "Failed to build server farm binary"
        return 1
    fi
    
    # Build consciousness binary
    cargo build --release --bin buhera-consciousness 2>&1 | tee "${LOGS_DIR}/consciousness_binary.log"
    if [ $? -eq 0 ]; then
        print_success "Consciousness binary built successfully"
    else
        print_error "Failed to build consciousness binary"
        return 1
    fi
    
    # Build monitor binary
    cargo build --release --bin buhera-monitor 2>&1 | tee "${LOGS_DIR}/monitor_binary.log"
    if [ $? -eq 0 ]; then
        print_success "Monitor binary built successfully"
    else
        print_error "Failed to build monitor binary"
        return 1
    fi
    
    print_success "All Rust components built successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    cd "${PROJECT_ROOT}"
    
    # Run unit tests
    print_status "Running unit tests..."
    cargo test --all-features 2>&1 | tee "${LOGS_DIR}/unit_tests.log"
    
    if [ $? -eq 0 ]; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        return 1
    fi
    
    # Run integration tests
    print_status "Running integration tests..."
    cargo test --all-features --test '*' 2>&1 | tee "${LOGS_DIR}/integration_tests.log"
    
    if [ $? -eq 0 ]; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        return 1
    fi
    
    # Run doc tests
    print_status "Running documentation tests..."
    cargo test --all-features --doc 2>&1 | tee "${LOGS_DIR}/doc_tests.log"
    
    if [ $? -eq 0 ]; then
        print_success "Documentation tests passed"
    else
        print_error "Documentation tests failed"
        return 1
    fi
    
    print_success "All tests completed successfully"
}

# Function to build documentation
build_documentation() {
    print_status "Building documentation..."
    
    cd "${PROJECT_ROOT}"
    
    # Build Rust documentation
    print_status "Building Rust documentation..."
    cargo doc --all-features --no-deps 2>&1 | tee "${LOGS_DIR}/documentation.log"
    
    if [ $? -eq 0 ]; then
        print_success "Documentation built successfully"
    else
        print_error "Failed to build documentation"
        return 1
    fi
    
    # Build additional documentation if needed
    print_status "Processing additional documentation..."
    
    # Check if we have LaTeX documents to build
    if [ -d "${PROJECT_ROOT}/docs/foundation" ]; then
        print_status "Building LaTeX documents..."
        
        cd "${PROJECT_ROOT}/docs/foundation"
        
        # Build LaTeX documents if pdflatex is available
        if command_exists pdflatex; then
            for tex_file in *.tex; do
                if [ -f "$tex_file" ]; then
                    print_status "Building $tex_file..."
                    pdflatex -interaction=nonstopmode "$tex_file" 2>&1 | tee "${LOGS_DIR}/latex_${tex_file%.tex}.log"
                    
                    # Run again for cross-references
                    pdflatex -interaction=nonstopmode "$tex_file" > /dev/null 2>&1
                fi
            done
            
            print_success "LaTeX documents built successfully"
        else
            print_warning "pdflatex not found, skipping LaTeX document generation"
        fi
    fi
    
    print_success "Documentation build completed"
}

# Function to create device tree files
create_device_tree() {
    print_status "Creating device tree files..."
    
    # Create gas oscillation device tree if it doesn't exist
    if [ ! -f "${PROJECT_ROOT}/boot/device-tree/gas-oscillation.dts" ]; then
        print_status "Creating gas oscillation device tree..."
        
        mkdir -p "${PROJECT_ROOT}/boot/device-tree"
        
        cat > "${PROJECT_ROOT}/boot/device-tree/gas-oscillation.dts" << 'EOF'
/dts-v1/;

/ {
    compatible = "buhera,gas-oscillation-server-farm";
    model = "Buhera VPOS Gas Oscillation Server Farm";
    
    gas-oscillation-chambers {
        compatible = "buhera,gas-oscillation-chamber";
        chamber-count = <1000>;
        pressure-range = <1 100>; /* 0.1 - 10.0 atm in deciseconds */
        temperature-range = <2000 4000>; /* 200 - 400 K in decikelvin */
        cycle-frequency = <1000>; /* Hz */
        
        molecular-density = <1000000000000000000000000000>; /* molecules/m³ */
        oscillation-sensitivity = <1>; /* 0.001 in milliunits */
        
        properties {
            phase-synchronization;
            amplitude-control;
            real-time-monitoring;
            processor-oscillator-duality;
            quantum-coherence-preservation;
            femtosecond-precision;
            molecular-computation;
            triple-function-mode;
        };
    };
    
    cooling-system {
        compatible = "buhera,zero-cost-cooling";
        circulation-rate = <1000>; /* m³/s */
        heat-recovery-efficiency = <95>; /* 95% */
        
        properties {
            zero-cost-cooling;
            entropy-prediction;
            optimal-atom-selection;
            adaptive-thermal-control;
            thermodynamic-inevitability;
            natural-cooling-preference;
            entropy-endpoint-cache;
            zero-energy-cooling;
        };
    };
    
    consciousness-substrate {
        compatible = "buhera,consciousness-substrate";
        substrate-type = "unified";
        memory-distribution = "distributed";
        coherence-threshold = <99>; /* 99% */
        
        properties {
            distributed-processing;
            adaptive-learning;
            pattern-recognition;
            temporal-integration;
            consciousness-networking;
            quantum-entangled-sync;
        };
    };
    
    virtual-foundry {
        compatible = "buhera,virtual-foundry";
        processor-creation-rate = <1000000000000>; /* processors/second */
        lifecycle-mode = "femtosecond";
        
        properties {
            infinite-processors;
            femtosecond-lifecycle;
            adaptive-specialization;
            unlimited-parallelization;
            virtual-memory-unlimited;
            processor-recycling;
        };
    };
    
    atomic-clock-network {
        compatible = "buhera,atomic-clock-network";
        precision-target = <1>; /* 1e-18 seconds in attoseconds */
        synchronization-protocol = "quantum-entangled";
        
        properties {
            coherence-maintenance;
            phase-noise-suppression;
            drift-compensation;
            quantum-entanglement-sync;
            stella-lorraine-mode;
        };
    };
    
    pressure-control {
        compatible = "buhera,pressure-control";
        control-mode = "adaptive";
        safety-limits = <5 150>; /* 0.05 - 15.0 atm in centiatm */
        
        properties {
            pressure-cycling;
            feedback-control;
            feedforward-control;
            pressure-prediction;
            safety-interlocks;
            emergency-shutdown;
        };
    };
    
    monitoring {
        compatible = "buhera,monitoring-system";
        metrics-collection-rate = <1000>; /* Hz */
        
        properties {
            real-time-monitoring;
            performance-logging;
            system-health-monitoring;
            predictive-maintenance;
            anomaly-detection;
            consciousness-monitoring;
        };
    };
};
EOF
        
        print_success "Gas oscillation device tree created"
    else
        print_status "Gas oscillation device tree already exists"
    fi
}

# Function to validate build
validate_build() {
    print_status "Validating build..."
    
    cd "${PROJECT_ROOT}"
    
    # Check if binaries exist
    if [ -f "${BUILD_DIR}/release/buhera-server-farm" ]; then
        print_success "Server farm binary exists"
    else
        print_error "Server farm binary not found"
        return 1
    fi
    
    if [ -f "${BUILD_DIR}/release/buhera-consciousness" ]; then
        print_success "Consciousness binary exists"
    else
        print_error "Consciousness binary not found"
        return 1
    fi
    
    if [ -f "${BUILD_DIR}/release/buhera-monitor" ]; then
        print_success "Monitor binary exists"
    else
        print_error "Monitor binary not found"
        return 1
    fi
    
    # Check if configuration files exist
    if [ -f "${PROJECT_ROOT}/etc/vpos/gas_oscillation.conf" ]; then
        print_success "Configuration file exists"
    else
        print_error "Configuration file not found"
        return 1
    fi
    
    # Test binary execution
    print_status "Testing binary execution..."
    
    # Test server farm binary
    if "${BUILD_DIR}/release/buhera-server-farm" --help > /dev/null 2>&1; then
        print_success "Server farm binary is executable"
    else
        print_error "Server farm binary is not executable"
        return 1
    fi
    
    # Test consciousness binary
    if "${BUILD_DIR}/release/buhera-consciousness" --help > /dev/null 2>&1; then
        print_success "Consciousness binary is executable"
    else
        print_error "Consciousness binary is not executable"
        return 1
    fi
    
    # Test monitor binary
    if "${BUILD_DIR}/release/buhera-monitor" --help > /dev/null 2>&1; then
        print_success "Monitor binary is executable"
    else
        print_error "Monitor binary is not executable"
        return 1
    fi
    
    print_success "Build validation completed successfully"
}

# Function to show build summary
show_build_summary() {
    print_status "Build Summary:"
    echo "============================================="
    echo "Project: Buhera VPOS Gas Oscillation Server Farm"
    echo "Build Directory: ${BUILD_DIR}"
    echo "Logs Directory: ${LOGS_DIR}"
    echo ""
    echo "Built Components:"
    echo "- Gas Oscillation Kernel Modules"
    echo "- Gas Oscillation Drivers"
    echo "- Server Farm Binary"
    echo "- Consciousness Binary"
    echo "- Monitor Binary"
    echo "- Documentation"
    echo "- Device Tree Files"
    echo ""
    echo "Available Binaries:"
    echo "- ${BUILD_DIR}/release/buhera-server-farm"
    echo "- ${BUILD_DIR}/release/buhera-consciousness"
    echo "- ${BUILD_DIR}/release/buhera-monitor"
    echo ""
    echo "Configuration:"
    echo "- ${PROJECT_ROOT}/etc/vpos/gas_oscillation.conf"
    echo ""
    echo "Next Steps:"
    echo "1. Review configuration file"
    echo "2. Run tests: cargo test --all-features"
    echo "3. Start server farm: ./target/release/buhera-server-farm start"
    echo "4. Monitor status: ./target/release/buhera-monitor"
    echo "============================================="
}

# Main build function
main() {
    local start_time=$(date +%s)
    
    print_status "Starting Buhera VPOS Gas Oscillation Server Farm build..."
    print_status "Build started at: $(date)"
    
    # Check dependencies
    check_dependencies
    
    # Create device tree files
    create_device_tree
    
    # Build kernel modules
    build_kernel_modules
    
    # Build Rust components
    build_rust_components
    
    # Run tests
    if [ "${SKIP_TESTS:-false}" != "true" ]; then
        run_tests
    else
        print_warning "Skipping tests (SKIP_TESTS=true)"
    fi
    
    # Build documentation
    if [ "${SKIP_DOCS:-false}" != "true" ]; then
        build_documentation
    else
        print_warning "Skipping documentation (SKIP_DOCS=true)"
    fi
    
    # Validate build
    validate_build
    
    local end_time=$(date +%s)
    local build_time=$((end_time - start_time))
    
    print_success "Build completed successfully in ${build_time} seconds!"
    print_status "Build finished at: $(date)"
    
    # Show build summary
    show_build_summary
}

# Handle command line arguments
case "${1:-}" in
    "kernel")
        check_dependencies
        build_kernel_modules
        ;;
    "rust")
        check_dependencies
        build_rust_components
        ;;
    "test")
        check_dependencies
        run_tests
        ;;
    "docs")
        check_dependencies
        build_documentation
        ;;
    "clean")
        print_status "Cleaning build artifacts..."
        cd "${PROJECT_ROOT}"
        cargo clean
        rm -rf "${LOGS_DIR}"
        print_success "Clean completed"
        ;;
    "help"|"-h"|"--help")
        echo "Buhera VPOS Gas Oscillation Server Farm Build Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (none)    - Build all components (default)"
        echo "  kernel    - Build only kernel modules"
        echo "  rust      - Build only Rust components"
        echo "  test      - Run tests only"
        echo "  docs      - Build documentation only"
        echo "  clean     - Clean build artifacts"
        echo "  help      - Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  SKIP_TESTS=true   - Skip running tests"
        echo "  SKIP_DOCS=true    - Skip building documentation"
        echo ""
        ;;
    *)
        # Default: build all
        main
        ;;
esac 