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
    print("  1. Quick demonstration (5 minutes)")
    print("  2. Compression validation (15 minutes)")
    print("  3. Network evolution validation (10 minutes)")
    print("  4. Foundry architecture validation (20 minutes)")
    print("  5. Virtual processing acceleration validation (15 minutes)")
    print("  6. Full validation suite (60 minutes)")
    print("  7. Exit")
    print()
    
    try:
        choice = input("Select option (1-7): ").strip()
        
        if choice == "1":
            print("\nRunning quick demonstration...")
            subprocess.run([sys.executable, "example_usage.py"], cwd=Path(__file__).parent)
        
        elif choice == "2":
            print("\nRunning compression validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--compression"], cwd=Path(__file__).parent)
        
        elif choice == "3":
            print("\nRunning network evolution validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--network-evolution"], cwd=Path(__file__).parent)
        
        elif choice == "4":
            print("\nRunning foundry architecture validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--foundry"], cwd=Path(__file__).parent)
        
        elif choice == "5":
            print("\nRunning virtual processing acceleration validation...")
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--virtual-acceleration"], cwd=Path(__file__).parent)
        
        elif choice == "6":
            print("\nRunning full validation suite...")
            output_dir = "validation_results"
            subprocess.run([sys.executable, "-m", "buhera_validation.cli", "--full-suite", "--output", output_dir], cwd=Path(__file__).parent)
            print(f"\nResults saved to: {output_dir}/")
        
        elif choice == "7":
            print("Goodbye!")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please select 1-7.")
            
    except KeyboardInterrupt:
        print("\n\nValidation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
