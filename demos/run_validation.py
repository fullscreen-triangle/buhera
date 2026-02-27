#!/usr/bin/env python3
"""
Quick Validation Runner

Simple script to run Buhera framework validation demonstrations.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run validation with simple interface."""
    
    print("=" * 60)
    print("BUHERA FRAMEWORK VALIDATION RUNNER")
    print("=" * 60)
    print()
    print("Available validation options:")
    print("  1. Quick demonstration (5 minutes) - SAVES JSON RESULTS")
    print("  2. Compression validation (15 minutes) - SAVES JSON RESULTS")
    print("  3. Network evolution validation (10 minutes) - SAVES JSON RESULTS")
    print("  4. Foundry architecture validation (20 minutes) - SAVES JSON RESULTS")
    print("  5. Virtual processing acceleration validation (15 minutes) - SAVES JSON RESULTS")
    print("  6. Proof-validated storage validation (25 minutes) - SAVES JSON RESULTS")
    print("  7. Alphabetical encoding validation (30 minutes) - SAVES JSON RESULTS")
    print("  8. Full validation suite (90 minutes) - SAVES JSON RESULTS")
    print("  9. Exit")
    print()
    
    try:
        choice = input("Select option (1-9): ").strip()
        
        # Create results directory
        output_dir = "validation_results"
        Path(output_dir).mkdir(exist_ok=True)
        
        if choice == "1":
            print("\n🔧 Running quick demonstration (FIXED IndexError + JSON results)...")
            subprocess.run([sys.executable, "example_usage_with_json.py"], cwd=Path(__file__).parent)
            print(f"\n💾 Results saved to: {output_dir}/")
        
        elif choice == "2":
            print("\nRunning compression validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--compression", "--output", output_dir], cwd=Path(__file__).parent)
            print(f"\n💾 Results saved to: {output_dir}/")
        
        elif choice == "3":
            print("\nRunning network evolution validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--network-evolution", "--output", output_dir], cwd=Path(__file__).parent)
            print(f"\n💾 Results saved to: {output_dir}/")
        
        elif choice == "4":
            print("\nRunning foundry architecture validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--foundry", "--output", output_dir], cwd=Path(__file__).parent)
            print(f"\n💾 Results saved to: {output_dir}/")
        
        elif choice == "5":
            print("\nRunning virtual processing acceleration validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--virtual-acceleration", "--output", output_dir], cwd=Path(__file__).parent)
            print(f"\n💾 Results saved to: {output_dir}/")
            
        elif choice == "6":
            print("\nRunning proof-validated storage validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--proof-storage", "--output", output_dir], cwd=Path(__file__).parent)
            print(f"\n💾 Results saved to: {output_dir}/")
            
        elif choice == "7":
            print("\n🔤 Running alphabetical encoding validation...")
            subprocess.run([sys.executable, "complete_alphabetical_validation.py"], cwd=Path(__file__).parent)
            print(f"\n💾 Results saved to: alphabetical_validation_results/")
        
        elif choice == "8":
            print("\nRunning full validation suite...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--full-suite", "--output", output_dir], cwd=Path(__file__).parent)
            print(f"\n💾 Results saved to: {output_dir}/")
        
        elif choice == "9":
            print("Goodbye!")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please select 1-9.")
            
    except KeyboardInterrupt:
        print("\n\nValidation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
