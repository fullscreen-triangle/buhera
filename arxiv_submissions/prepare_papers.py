#!/usr/bin/env python3
"""
ArXiv Paper Preparation Script
Automates the preparation of Buhera research papers for arXiv submission
"""

import os
import shutil
import re
from pathlib import Path

class ArxivPaperPreparator:
    def __init__(self):
        self.base_dir = Path(".")
        self.docs_dir = Path("../docs")
        self.submissions_dir = Path("arxiv_submissions")
        
        # Paper mappings: (source_path, arxiv_name, primary_category, secondary_categories)
        self.papers = [
            ("buhera-vpos.tex", "buhera_vpos_arxiv.tex", "quant-ph", ["cs.ET", "physics.comp-ph"]),
            ("foundation/kwasa-kwasa.tex", "kwasa_kwasa_arxiv.tex", "cs.AI", ["cs.CL", "q-bio.NC"]),
            ("theory/zero-lag-information-transfer.tex", "zero_lag_information_arxiv.tex", "quant-ph", ["physics.optics", "cs.IT"]),
            ("foundation/problem-reduction.tex", "problem_reduction_arxiv.tex", "q-bio.NC", ["quant-ph", "cs.AI"]),
            ("theory/vpos-theory.tex", "vpos_theory_arxiv.tex", "cs.OS", ["quant-ph", "cs.AI"]),
            ("foundation/mass-spec.tex", "mass_spec_arxiv.tex", "physics.chem-ph", ["physics.ins-det", "physics.data-an"]),
            ("foundation/bio-oscillations.tex", "bio_oscillations_arxiv.tex", "q-bio.NC", ["physics.bio-ph", "nlin.AO"]),
            ("foundation/oscillatory-theorem.tex", "oscillatory_theorem_arxiv.tex", "math-ph", ["nlin.PS", "physics.gen-ph"]),
            ("theory/land-speed-record.tex", "land_speed_record_arxiv.tex", "physics.flu-dyn", ["physics.class-ph", "physics.app-ph"]),
            ("theory/zero-time-travel.tex", "zero_time_travel_arxiv.tex", "gr-qc", ["quant-ph", "physics.optics"])
        ]
        
        self.standard_author = """Kundai Farai Sachikonye\\\\
Independent Research\\\\
Buhera Framework Project\\\\
Zimbabwe\\\\
\\texttt{kundai.sachikonye@wzw.tum.de}"""

    def clean_latex_content(self, content):
        """Clean up LaTeX content for arXiv submission"""
        # Remove or standardize problematic packages
        content = re.sub(r'\\usepackage\{fancyhdr\}.*?\n', '', content)
        content = re.sub(r'\\pagestyle\{fancy\}.*?\n', '', content)
        content = re.sub(r'\\fancyhf\{\}.*?\n', '', content)
        content = re.sub(r'\\fancyhead.*?\n', '', content)
        content = re.sub(r'\\rhead.*?\n', '', content)
        content = re.sub(r'\\lhead.*?\n', '', content)
        
        # Standardize geometry
        content = re.sub(r'\\geometry\{.*?\}', r'\\geometry{margin=1in}', content)
        
        # Clean up excessive spacing
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content

    def add_arxiv_header(self, content, primary_cat, secondary_cats):
        """Add arXiv category header to the document"""
        header = f"""% ArXiv category declaration
% Primary: {primary_cat}
% Secondary: {', '.join(secondary_cats)}

"""
        
        # Insert after documentclass but before other packages
        lines = content.split('\n')
        insert_pos = 1
        for i, line in enumerate(lines):
            if line.startswith('\\documentclass'):
                insert_pos = i + 1
                break
        
        lines.insert(insert_pos, header.strip())
        return '\n'.join(lines)

    def standardize_author(self, content):
        """Standardize author information"""
        # Find and replace author block
        author_pattern = r'\\author\{.*?\}'
        author_replacement = f'\\author{{\n{self.standard_author}\n}}'
        
        content = re.sub(author_pattern, author_replacement, content, flags=re.DOTALL)
        return content

    def add_keywords(self, content, keywords):
        """Add keywords to abstract if not present"""
        if "textbf{Keywords" not in content:
            # Find end of abstract
            abstract_end = content.find('\\end{abstract}')
            if abstract_end != -1:
                keywords_text = f"\n\n\\textbf{{Keywords:}} {keywords}\n"
                content = content[:abstract_end] + keywords_text + content[abstract_end:]
        return content

    def prepare_paper(self, source_path, arxiv_name, primary_cat, secondary_cats):
        """Prepare a single paper for arXiv submission"""
        full_source_path = self.docs_dir / source_path
        
        if not full_source_path.exists():
            print(f"Warning: Source file {full_source_path} not found")
            return False
        
        try:
            with open(full_source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply transformations
            content = self.clean_latex_content(content)
            content = self.add_arxiv_header(content, primary_cat, secondary_cats)
            content = self.standardize_author(content)
            
            # Add default keywords if none present
            default_keywords = self.get_default_keywords(primary_cat)
            content = self.add_keywords(content, default_keywords)
            
            # Write to arxiv submissions directory
            output_path = self.submissions_dir / arxiv_name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Prepared: {arxiv_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error preparing {arxiv_name}: {e}")
            return False

    def get_default_keywords(self, primary_cat):
        """Get default keywords based on primary category"""
        keyword_map = {
            "quant-ph": "quantum computing, quantum information, quantum physics",
            "cs.AI": "artificial intelligence, machine learning, semantic processing",
            "cs.OS": "operating systems, computer systems, system architecture", 
            "q-bio.NC": "neuroscience, cognition, biological computation",
            "physics.chem-ph": "chemical physics, molecular dynamics, computational chemistry",
            "math-ph": "mathematical physics, theoretical physics, applied mathematics",
            "physics.flu-dyn": "fluid dynamics, aerodynamics, computational fluid dynamics",
            "gr-qc": "general relativity, quantum gravity, spacetime physics",
            "physics.bio-ph": "biological physics, biophysics, computational biology"
        }
        return keyword_map.get(primary_cat, "theoretical physics, computational science")

    def compile_paper(self, arxiv_name):
        """Attempt to compile a paper to check for errors"""
        paper_path = self.submissions_dir / arxiv_name
        if not paper_path.exists():
            return False
        
        try:
            import subprocess
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', arxiv_name],
                cwd=self.submissions_dir,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            print(f"⚠️  Could not compile {arxiv_name} (pdflatex not available)")
            return None

    def create_submission_checklist(self):
        """Create a submission checklist for each paper"""
        checklist_content = """# ArXiv Submission Checklist

## Pre-Submission Verification

### For each paper, verify:
- [ ] LaTeX compiles without errors
- [ ] Abstract under 1920 characters
- [ ] Title under 200 characters  
- [ ] Author information standardized
- [ ] Keywords present and relevant
- [ ] Bibliography properly formatted
- [ ] Equations and math notation correct
- [ ] Figures (if any) compile correctly

## Papers Ready for Submission:

"""
        
        for source_path, arxiv_name, primary_cat, secondary_cats in self.papers:
            checklist_content += f"""
### {arxiv_name}
- **Primary Category**: {primary_cat}
- **Secondary Categories**: {', '.join(secondary_cats)}
- **Source**: {source_path}
- **Status**: [ ] Ready for upload
- **ArXiv ID**: _______________
- **Submission Date**: _______________

"""

        checklist_content += """
## Submission Order Recommendation:

1. **Phase 1 (Foundational)**: buhera_vpos_arxiv.tex, oscillatory_theorem_arxiv.tex
2. **Phase 2 (Information Theory)**: zero_lag_information_arxiv.tex, zero_time_travel_arxiv.tex  
3. **Phase 3 (AI/Computation)**: kwasa_kwasa_arxiv.tex, vpos_theory_arxiv.tex
4. **Phase 4 (Applied)**: problem_reduction_arxiv.tex, bio_oscillations_arxiv.tex, mass_spec_arxiv.tex, land_speed_record_arxiv.tex

## Notes:
- Submit foundational papers first to establish theoretical basis
- Wait for acceptance before submitting dependent papers
- Consider cross-referencing between related papers
- Update references as papers receive arXiv IDs
"""
        
        checklist_path = self.submissions_dir / "submission_checklist.md"
        with open(checklist_path, 'w', encoding='utf-8') as f:
            f.write(checklist_content)
        
        print("📋 Created submission checklist")

    def run_preparation(self):
        """Run the complete paper preparation process"""
        print("🚀 Starting ArXiv paper preparation...")
        
        # Create submissions directory if it doesn't exist
        self.submissions_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_papers = len(self.papers)
        
        for source_path, arxiv_name, primary_cat, secondary_cats in self.papers:
            if self.prepare_paper(source_path, arxiv_name, primary_cat, secondary_cats):
                success_count += 1
                
                # Try to compile if possible
                compile_result = self.compile_paper(arxiv_name)
                if compile_result is True:
                    print(f"  ✅ Compilation successful")
                elif compile_result is False:
                    print(f"  ⚠️  Compilation failed - check manually")
        
        # Create submission checklist
        self.create_submission_checklist()
        
        print(f"\n📊 Preparation Summary:")
        print(f"   Successfully prepared: {success_count}/{total_papers} papers")
        print(f"   Output directory: {self.submissions_dir}")
        print(f"   Next step: Review papers and begin arXiv submission")

if __name__ == "__main__":
    preparator = ArxivPaperPreparator()
    preparator.run_preparation() 