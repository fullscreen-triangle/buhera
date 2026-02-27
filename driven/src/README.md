# Buhera OS Validation Framework

Comprehensive validation suite for the Buhera Operating System claims.

## Overview

This framework validates the three major claims from the Buhera OS paper:

1. **Sorting Performance**: 10^6× speedup via O(log₃ N) categorical navigation
2. **IPC Performance**: 10^2× speedup via categorical address sharing
3. **Commutation Relations**: [Ô_cat, Ô_phys] = 0 for zero-cost demon operations

## Structure

```
driven/src/
├── core.py                      # Core categorical computing primitives
├── run_all_validations.py       # Master validation runner
├── sorting/
│   └── validate_sorting.py      # Sorting performance validation
├── ipc/
│   └── validate_ipc.py          # IPC performance validation
└── commutation/
    └── validate_commutation.py  # Commutation relation validation
```

## Quick Start

### Run All Validations (Standard Mode)

```bash
cd driven/src
python run_all_validations.py
```

Results saved to `driven/data/master_validation_report_TIMESTAMP.json`

### Run Individual Validations

**Sorting:**
```bash
python sorting/validate_sorting.py
```

**IPC:**
```bash
python ipc/validate_ipc.py
```

**Commutation:**
```bash
python commutation/validate_commutation.py
```

### Validation Modes

- **Quick Mode** (~2-5 minutes):
  ```bash
  python run_all_validations.py --quick
  ```

- **Standard Mode** (~10-20 minutes):
  ```bash
  python run_all_validations.py --standard
  ```

- **Full Mode** (~30-60 minutes):
  ```bash
  python run_all_validations.py --full
  ```

## Requirements

```bash
pip install numpy scipy
```

No other dependencies required.

## What Gets Validated

### 1. Sorting Performance

- **Complexity Scaling**: Validates O(log₃ N) vs O(N log N)
- **Speedup Trends**: Measures how speedup scales with problem size
- **Energy Efficiency**: Compares energy consumption
- **Correctness**: Verifies results are identical
- **Extrapolation**: Predicts performance at N=10^6, 10^7, 10^8

**Key Metrics:**
- Operations count (categorical vs conventional)
- Wall-clock time
- Energy dissipation
- Speedup ratio
- R² fit quality for complexity curves

### 2. IPC Performance

- **Latency Scaling**: Should be constant regardless of data size
- **Zero-Copy Validation**: Confirms no data copying occurs
- **Speedup vs Mechanisms**: Compares against pipe, shared memory, message queue
- **Energy Efficiency**: Validates dramatically lower energy cost

**Key Metrics:**
- Latency (ms)
- Bandwidth (GB/s)
- Number of copies
- Navigation steps
- Speedup ratio per mechanism

### 3. Commutation Relations

- **Position Operators**: [n̂, x̂], [l̂, x̂], [m̂, x̂]
- **Momentum Operators**: [n̂, p̂], [l̂, p̂], [m̂, p̂]
- **Hamiltonian**: [n̂, Ĥ], [l̂, Ĥ], [m̂, Ĥ]
- **Self-Commutation**: [n̂, l̂], [n̂, m̂], [l̂, m̂]
- **Deviation Analysis**: Why momentum shows small deviation

**Key Metrics:**
- Commutator norm
- Relative commutator
- Hilbert space dimension dependence
- Scaling with truncation

## Results Format

All results saved as JSON in `driven/data/`:

```json
{
  "timestamp": "2025-12-12T19:30:45",
  "configuration": {...},
  "benchmarks": [...],
  "summary": {
    "overall_mean_speedup": 8.45,
    "best_speedup": 13.4,
    "claim_validation": {
      "categorical_is_log3_n": true,
      "speedup_increases_with_n": true,
      "energy_dramatically_lower": true
    }
  }
}
```

## Understanding Results

### Sorting Claims

- ✓ **PASS**: R² > 0.95 for O(log₃ N) fit, speedup increasing with N
- ✗ **FAIL**: Poor fit quality or speedup decreasing

### IPC Claims

- ✓ **PASS**: Speedup > 100× for large data, latency constant
- ✗ **FAIL**: Speedup < 100× or latency scales with data size

### Commutation Claims

- ✓ **PASS**: Commutator norm < 10⁻¹⁰ (numerical zero)
- ⚠ **PARTIAL**: Small but nonzero commutator (finite truncation)
- ✗ **FAIL**: Large commutator that doesn't vanish

## Interpreting Speedup

**Current Results:**
- N=100: ~1× (overhead dominates)
- N=1,000: ~5×
- N=10,000: ~13×
- N=100,000: ~40× (extrapolated)
- N=1,000,000: ~350× (extrapolated)
- N=10,000,000: ~3,500× (extrapolated)

**Note**: The 10^6× claim is asymptotic. Current validations reach ~13× at N=10⁴.
Extrapolation suggests 10^6× requires N ≈ 10⁹.

## Extending the Framework

### Add New Validation

1. Create module in new folder: `driven/src/new_validation/`
2. Implement validation function returning Dict[str, Any]
3. Save results using `core.save_results()`
4. Add to `run_all_validations.py`

### Example:

```python
from core import save_results

def validate_new_claim():
    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": [...],
        "summary": {...}
    }

    save_results(results, "new_validation_TIMESTAMP.json")
    return results
```

## Troubleshooting

**ImportError:**
```bash
# Make sure you're in driven/src/ directory
cd driven/src
python run_all_validations.py
```

**MemoryError:**
```bash
# Use quick mode for limited RAM
python run_all_validations.py --quick
```

**Slow Performance:**
```bash
# Reduce n_trials or problem sizes in individual scripts
```

## Citation

If you use this validation framework, please cite:

```bibtex
@article{buhera-os-2025,
  title={Buhera: A Categorical Operating System Based on Trajectory Completion},
  author={[Author Names]},
  journal={[Journal]},
  year={2025}
}
```

## License

[Specify license]
