import React, { useState } from "react";
import Modal from "react-modal";

export default function News({ ActiveIndex }) {
  const [isOpen, setIsOpen] = useState(false);
  const [modalContent, setModalContent] = useState({});

  const papers = [
    {
      num: "01",
      title: "Buhera: A Categorical Operating System Based on Trajectory Completion",
      journal: "Research Paper",
      status: "In Preparation",
      statusColor: "#f59e0b",
      abstract: "We present Buhera, an operating system architecture that operates by trajectory completion rather than forward simulation. Processes specify desired final states as categorical addresses in partition space, and the system navigates backward to identify penultimate states from which completion requires minimal operations.",
      highlights: [
        "O(log\u2083 N) sorting complexity with R\u00B2 = 1.000",
        "55x speedup at N = 10,000",
        "9/9 commutation relations validated < 10\u207B\u00B9\u2070",
        "Energy consumption reduced to 6% of conventional",
        "Penultimate state detection with 100% success rate"
      ],
      sections: "Mathematical Foundations, Categorical Computing Architecture, OS Design, Experimental Validation, Performance Analysis"
    },
    {
      num: "02",
      title: "vaHera: A Categorical Scripting Language with Compile-Time Complexity Verification",
      journal: "Research Paper",
      status: "In Preparation",
      statusColor: "#f59e0b",
      abstract: "We present vaHera, a domain-specific scripting language designed for the Buhera operating system. vaHera enforces categorical correctness through dependent types and proves complexity bounds at compile time.",
      highlights: [
        "Dependent type system for complexity verification",
        "Native S-coordinate and ternary address types",
        "Compile-time complexity bound checking",
        "Zero-knowledge proof integration",
        "Built on Rust for memory safety"
      ],
      sections: "Language Design, Type System, Compiler Architecture, Runtime, Standard Library"
    },
    {
      num: "03",
      title: "Zero-Cost Demon Operations via Categorical-Physical Commutation",
      journal: "Planned",
      status: "Planned",
      statusColor: "#6366f1",
      abstract: "We demonstrate that categorical sorting operations commute with all physical observables, enabling thermodynamically free information processing. This resolves the Maxwell's demon paradox for categorical operations.",
      highlights: [
        "Formal proof of categorical-physical commutation",
        "Hydrogen atom basis validation (dim 56)",
        "Finite size scaling analysis",
        "Implications for thermodynamic computing",
        "Connection to Landauer's principle"
      ],
      sections: "Quantum Operator Theory, Commutation Proofs, Experimental Validation, Thermodynamic Analysis"
    }
  ];

  return (
    <>
      <div
        className={
          ActiveIndex === 3
            ? "cavani_tm_section active animated fadeInUp"
            : "cavani_tm_section hidden animated"
        }
        id="news__"
      >
        <div className="section_inner">
          <div className="cavani_tm_news">
            <div className="cavani_tm_title">
              <span>Publications</span>
            </div>
            <p className="buhera_text" style={{ marginBottom: "30px" }}>
              The Buhera framework is documented across multiple research papers covering the operating system design,
              the vaHera scripting language, and the underlying mathematical theory.
            </p>

            <div className="buhera_papers_list">
              {papers.map((p, i) => (
                <div key={i} className="buhera_paper_card" onClick={() => { setModalContent(p); setIsOpen(true); }}>
                  <div className="paper_num">{p.num}</div>
                  <div className="paper_content">
                    <div className="paper_meta">
                      <span className="paper_journal">{p.journal}</span>
                      <span className="paper_status" style={{ color: p.statusColor, borderColor: p.statusColor }}>{p.status}</span>
                    </div>
                    <h3 className="paper_title">{p.title}</h3>
                    <p className="paper_abstract">{p.abstract}</p>
                    <div className="paper_highlights">
                      {p.highlights.slice(0, 3).map((h, j) => (
                        <span key={j} className="highlight_tag">{h}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Research Figures Gallery */}
            <div className="buhera_block" style={{ marginTop: "40px" }}>
              <h3 className="buhera_subtitle">Research Figures</h3>
              <div className="buhera_figures_gallery" style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "15px",
                marginTop: "20px"
              }}>
                {[
                  { src: "img/buhera/figure_sorting.png", cap: "Sorting Performance" },
                  { src: "img/buhera/figure_commutation.png", cap: "Commutation Relations" },
                  { src: "img/buhera/figure_partition_tree.png", cap: "Partition Tree Architecture" },
                  { src: "img/buhera/figure_s_coordinates.png", cap: "S-Entropy Coordinates" },
                  { src: "img/buhera/figure_processor.png", cap: "Categorical Processor" },
                ].map((fig, i) => (
                  <div key={i} className="buhera_fig_item" style={{
                    borderRadius: "8px",
                    overflow: "hidden",
                    border: "1px solid rgba(14, 165, 233, 0.15)",
                    cursor: "pointer",
                    transition: "border-color 0.3s"
                  }}>
                    <img src={fig.src} alt={fig.cap} style={{ width: "100%", display: "block" }} />
                    <div style={{
                      padding: "10px 12px",
                      background: "rgba(14, 165, 233, 0.05)",
                      fontSize: "12px",
                      color: "#8899a6",
                      textAlign: "center",
                      letterSpacing: "1px"
                    }}>{fig.cap}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {modalContent.title && (
        <Modal isOpen={isOpen} onRequestClose={() => setIsOpen(false)} className="mymodal" overlayClassName="myoverlay" closeTimeoutMS={300}>
          <div className="cavani_tm_modalbox opened">
            <div className="box_inner">
              <div className="close" onClick={() => setIsOpen(false)}>
                <a href="#"><i className="icon-cancel"></i></a>
              </div>
              <div className="description_wrap">
                <div className="service_popup_informations">
                  <div className="details">
                    <span style={{ color: modalContent.statusColor }}>{modalContent.status}</span>
                    <h3 style={{ color: "#e2e8f0", fontSize: "18px", marginTop: "8px" }}>{modalContent.title}</h3>
                  </div>
                  <div className="descriptions" style={{ marginTop: "20px" }}>
                    <p style={{ color: "#a0aab4" }}><strong style={{ color: "#8ec8f0" }}>Abstract:</strong> {modalContent.abstract}</p>
                    <p style={{ color: "#a0aab4", marginTop: "15px" }}><strong style={{ color: "#8ec8f0" }}>Key Results:</strong></p>
                    <ul style={{ listStyle: "none", padding: 0, margin: "10px 0" }}>
                      {modalContent.highlights && modalContent.highlights.map((h, j) => (
                        <li key={j} style={{ color: "#a0aab4", padding: "5px 0", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                          <span style={{ color: "#10b981", marginRight: "8px" }}>&#10003;</span> {h}
                        </li>
                      ))}
                    </ul>
                    <p style={{ color: "#64748b", fontSize: "12px", marginTop: "15px" }}>
                      <strong>Sections:</strong> {modalContent.sections}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Modal>
      )}
    </>
  );
}
