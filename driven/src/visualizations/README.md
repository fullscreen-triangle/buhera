# Buhera OS Visualization Suite

Professional publication-quality figures for the Buhera Operating System papers.

## Overview

This suite generates 5 comprehensive 4-panel figures covering all major aspects of the Buhera OS framework:

1. **Figure 1: Categorical Sorting Performance**
   - Complexity scaling (O(log₃ N) validation)
   - Speedup vs problem size
   - Energy efficiency landscape (3D)
   - Speedup matrix across distributions

2. **Figure 2: Categorical-Physical Commutation**
   - Commutator magnitude heatmap
   - Validation scatter plot
   - Operator space visualization (3D)
   - Finite size scaling

3. **Figure 3: Ternary Partition Trees**
   - Tree structure diagram
   - Navigation complexity comparison
   - 3D partition space
   - Capacity scaling

4. **Figure 4: S-Entropy Coordinate Addressing**
   - 2D S-coordinate space
   - Address resolution trajectory
   - 3D S-entropy space (Sk, St, Se)
   - Entropy composition by system type

5. **Figure 5: Categorical Processor**
   - Processing pipeline
   - Completion vs simulation
   - 3D state space trajectories
   - Penultimate state detection

## Quick Start

### Generate All Figures

```bash
cd driven/src/visualizations
python generate_all_figures.py
```

Figures saved to: `driven/figures/`

### Generate Individual Figures

```bash
python figure_sorting.py
python figure_commutation.py
python figure_partition_tree.py
python figure_s_coordinates.py
python figure_processor.py
```

## Output Formats

Each figure is saved in TWO formats:
- **PDF** (vector graphics) - Use in LaTeX papers
- **PNG** (300 DPI raster) - Use in presentations

## Figure Specifications

### Layout
- 4 panels per figure (horizontal layout)
- Figure size: 16" × 3.5"
- Font: Serif (LaTeX compatible)
- Resolution: 300 DPI

### Color Scheme
- Primary: #2E86AB (Blue)
- Secondary: #F18F01 (Orange)
- Tertiary: #A23B72 (Purple)
- Accent: #E63946 (Red)

All colors are colorblind-friendly.

### Panel Requirements
- Minimal text (as requested)
- At least one 3D plot per figure
- No text-based/table charts
- Publication-ready quality

## Using in LaTeX Papers

### Method 1: Include All Figures

```latex
\input{figures/figure_includes.tex}
```

This automatically includes all 5 figures with captions and labels.

### Method 2: Include Individual Figures

```latex
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/figure_sorting.pdf}
    \caption{Your caption here.}
    \label{fig:sorting}
\end{figure*}
```

### Required LaTeX Packages

```latex
\usepackage{graphicx}
\usepackage{subfig}  % If using subfigures
```

## Customization

### Change Output Directory

```bash
python generate_all_figures.py --output /path/to/output
```

### Modify Individual Figures

Each figure script is standalone and can be edited:

```python
# Example: Change colors in figure_sorting.py
cat_color = '#YOUR_HEX_COLOR'
conv_color = '#YOUR_HEX_COLOR'
```

### Adjust Figure Size

```python
# In any figure script
fig = plt.figure(figsize=(16, 3.5))  # Width, Height in inches
```

## Dependencies

```bash
pip install numpy matplotlib
```

No additional dependencies required!

## Figure Details

### Figure 1: Sorting
- **Panel (a)**: Log-log complexity plot with fit lines
- **Panel (b)**: Speedup scaling with asymptotic trend
- **Panel (c)**: 3D energy surface (N × trials)
- **Panel (d)**: Heatmap of speedup across distributions

**Key Results**:
- R² = 1.000 for O(log₃ N) fit
- 55× speedup at N=10⁴
- Energy ratio: 0.06

### Figure 2: Commutation
- **Panel (a)**: 3×3 commutator matrix heatmap
- **Panel (b)**: Scatter plot of 9 validation tests
- **Panel (c)**: 3D operator space with commutation links
- **Panel (d)**: Scaling with Hilbert space dimension

**Key Results**:
- All commutators < 10⁻¹⁰
- 9/9 tests pass
- Vanishes as n⁻²

### Figure 3: Partition Trees
- **Panel (a)**: Ternary tree with depth=3
- **Panel (b)**: Complexity comparison (log₃ vs log₂ vs linear)
- **Panel (c)**: 3D partition cells with color-coded branches
- **Panel (d)**: Capacity scaling (ternary vs binary vs quantum)

**Key Results**:
- 37% fewer steps than binary
- 3ᵈ leaves at depth d
- Hierarchical O(log₃ N) navigation

### Figure 4: S-Coordinates
- **Panel (a)**: 2D S-coordinate space with ternary grid
- **Panel (b)**: Address resolution path (5 steps)
- **Panel (c)**: 3D S-entropy space (Sₖ, Sₜ, Sₑ)
- **Panel (d)**: Entropy composition by system type

**Key Results**:
- 9 partition regions per plane
- Hierarchical refinement
- System-specific entropy signatures

### Figure 5: Processor
- **Panel (a)**: 5-stage pipeline with complexity labels
- **Panel (b)**: Time comparison (completion vs simulation)
- **Panel (c)**: 3D trajectories with penultimate state
- **Panel (d)**: Distance convergence plot

**Key Results**:
- O(log₃ N) + O(1) total complexity
- Single completion morphism
- Penultimate state detection

## File Structure

```
driven/src/visualizations/
├── README.md                    # This file
├── __init__.py
├── generate_all_figures.py      # Master generation script
├── figure_sorting.py            # Figure 1
├── figure_commutation.py        # Figure 2
├── figure_partition_tree.py     # Figure 3
├── figure_s_coordinates.py      # Figure 4
└── figure_processor.py          # Figure 5

driven/figures/                   # Output directory
├── figure_sorting.pdf
├── figure_sorting.png
├── figure_commutation.pdf
├── figure_commutation.png
├── figure_partition_tree.pdf
├── figure_partition_tree.png
├── figure_s_coordinates.pdf
├── figure_s_coordinates.png
├── figure_processor.pdf
├── figure_processor.png
└── figure_includes.tex          # LaTeX includes
```

## Quality Checklist

All figures include:
- [x] 4 panels in horizontal layout
- [x] At least one 3D visualization
- [x] Minimal text labels
- [x] No tables or text-heavy charts
- [x] Publication-quality resolution (300 DPI)
- [x] Vector format (PDF) for papers
- [x] Colorblind-friendly palette
- [x] Grid lines and proper axes
- [x] Legends with transparency
- [x] Consistent styling across all figures

## Troubleshooting

**Import Error**: Make sure you're running from `driven/src/visualizations/`

```bash
cd driven/src/visualizations
python generate_all_figures.py
```

**Display Issues**: If figures don't show, add at end of script:

```python
plt.show()
```

**Memory Issues**: Generate figures one at a time instead of all at once.

## Citation

If you use these visualizations in your work, please cite:

```bibtex
@article{buhera-os-2025,
  title={Buhera: A Categorical Operating System Based on Trajectory Completion},
  author={[Authors]},
  year={2025}
}
```

## License

[Specify license]
