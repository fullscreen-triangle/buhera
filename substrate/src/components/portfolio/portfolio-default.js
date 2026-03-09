import React, { useState } from "react";
import Modal from "react-modal";

export default function PortfolioDefault({ ActiveIndex }) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedFig, setSelectedFig] = useState(null);

  const results = [
    {
      img: "img/buhera/figure_sorting.png",
      title: "Sorting Complexity Validation",
      category: "Performance",
      metrics: [
        { label: "Complexity", value: "O(log\u2083 N)", status: "validated" },
        { label: "R\u00B2 Fit", value: "1.000", status: "validated" },
        { label: "Speedup (N=10k)", value: "55x", status: "validated" },
        { label: "Energy Ratio", value: "6%", status: "validated" },
      ],
      description: "Log-log regression confirms categorical sorting follows O(log\u2083 N) with perfect fit quality. Speedups increase monotonically with N, from 3.5x at N=100 to 55x at N=10,000, with extrapolated 170,000x at N=10\u2078."
    },
    {
      img: "img/buhera/figure_commutation.png",
      title: "Commutation Relations",
      category: "Physics",
      metrics: [
        { label: "Tests Passed", value: "9/9", status: "validated" },
        { label: "Max Residual", value: "< 10\u207B\u00B9\u2070", status: "validated" },
        { label: "Scaling", value: "n\u207B\u00B2", status: "validated" },
        { label: "Basis Dim", value: "56", status: "info" },
      ],
      description: "All nine categorical-physical operator commutators vanish within numerical precision. Finite size scaling confirms residuals decrease as n_max\u207B\u00B2, proving exact commutation in the infinite-dimensional limit."
    },
    {
      img: "img/buhera/figure_partition_tree.png",
      title: "Partition Tree Architecture",
      category: "Structure",
      metrics: [
        { label: "Nav Improvement", value: "37%", status: "validated" },
        { label: "Capacity (d=20)", value: "3.5\u00D710\u2079", status: "info" },
        { label: "Nav Steps", value: "log\u2083 N", status: "validated" },
        { label: "Step Accuracy", value: "\u00B11", status: "validated" },
      ],
      description: "Ternary trees require 37% fewer levels than binary for the same number of leaves. Navigation step counts match theoretical predictions across all tested sizes with 1000 random trials per N."
    },
    {
      img: "img/buhera/figure_s_coordinates.png",
      title: "S-Entropy Addressing",
      category: "Memory",
      metrics: [
        { label: "Mean Error", value: "1.5\u00D710\u207B\u2075", status: "validated" },
        { label: "Resolution (d=10)", value: "3\u207B\u00B9\u2070", status: "info" },
        { label: "Convergence", value: "3x/level", status: "validated" },
        { label: "5-Level Error", value: "< 0.1%", status: "validated" },
      ],
      description: "S-coordinate encoding is bijective up to discretization. Each additional refinement level reduces error by factor ~3. Different physical systems occupy distinct, well-separated regions in S-space."
    },
    {
      img: "img/buhera/figure_processor.png",
      title: "Processor Operation",
      category: "Execution",
      metrics: [
        { label: "Detection Rate", value: "100%", status: "validated" },
        { label: "Completion Cost", value: "O(1)", status: "validated" },
        { label: "Steps (N=1k)", value: "7.2\u00B10.8", status: "validated" },
        { label: "Speedup (N=10\u2076)", value: "77,000x", status: "projected" },
      ],
      description: "Penultimate state detection achieves 100% success rate. Completion morphism is consistently O(1), exactly 1 operation independent of N. Categorical distance converges geometrically at each navigation step."
    }
  ];

  return (
    <>
      <div
        className={
          ActiveIndex === 2
            ? "cavani_tm_section active animated fadeInUp"
            : "cavani_tm_section hidden animated"
        }
        id="portfolio_"
      >
        <div className="section_inner">
          <div className="cavani_tm_portfolio">
            <div className="cavani_tm_title">
              <span>Experimental Results</span>
            </div>
            <p className="buhera_text" style={{ marginBottom: "30px" }}>
              Comprehensive validation confirms all theoretical claims. Experiments conducted on standard hardware (3.0 GHz CPU, 16 GB RAM) using Python demonstrate categorical advantages even in software simulation.
            </p>

            {/* Summary Stats Banner */}
            <div className="buhera_results_banner" style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "15px",
              marginBottom: "35px"
            }}>
              <div className="buhera_stat_card">
                <span className="stat_value" style={{ color: "#0ea5e9" }}>7/7</span>
                <span className="stat_label">Claims Validated</span>
              </div>
              <div className="buhera_stat_card">
                <span className="stat_value" style={{ color: "#10b981" }}>R&sup2;=1.0</span>
                <span className="stat_label">Perfect Fit</span>
              </div>
              <div className="buhera_stat_card">
                <span className="stat_value" style={{ color: "#a78bfa" }}>55x</span>
                <span className="stat_label">Measured Speedup</span>
              </div>
              <div className="buhera_stat_card">
                <span className="stat_value" style={{ color: "#f59e0b" }}>9/9</span>
                <span className="stat_label">Commutation Tests</span>
              </div>
            </div>

            {/* Results Grid */}
            <div className="buhera_results_grid">
              {results.map((r, i) => (
                <div key={i} className="buhera_result_card" onClick={() => { setSelectedFig(r); setIsOpen(true); }}>
                  <div className="result_image" style={{
                    backgroundImage: `url(${r.img})`,
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    height: "180px",
                    borderRadius: "6px 6px 0 0",
                    borderBottom: "1px solid rgba(14, 165, 233, 0.15)"
                  }} />
                  <div className="result_content">
                    <span className="result_category">{r.category}</span>
                    <h4 className="result_title">{r.title}</h4>
                    <div className="result_metrics">
                      {r.metrics.map((m, j) => (
                        <div key={j} className="metric_item">
                          <span className="metric_label">{m.label}</span>
                          <span className={`metric_value ${m.status}`}>{m.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Validation Summary Table */}
            <div className="buhera_block" style={{ marginTop: "35px" }}>
              <h3 className="buhera_subtitle">Validation Summary</h3>
              <div className="buhera_table_wrap">
                <table className="buhera_table">
                  <thead>
                    <tr>
                      <th>Claim</th>
                      <th>Theoretical</th>
                      <th>Measured</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Sorting Complexity</td><td>O(log&#8323; N)</td><td>R&sup2; = 1.000</td><td><span className="status_badge validated">Validated</span></td></tr>
                    <tr><td>Speedup Scaling</td><td>Increases with N</td><td>3.5x to 55x</td><td><span className="status_badge validated">Validated</span></td></tr>
                    <tr><td>Commutation</td><td>[O_cat, O_phys] = 0</td><td>&lt; 10&#8315;&#185;&#8304; (9/9)</td><td><span className="status_badge validated">Validated</span></td></tr>
                    <tr><td>Navigation</td><td>37% vs binary</td><td>30-37%</td><td><span className="status_badge validated">Validated</span></td></tr>
                    <tr><td>Energy Efficiency</td><td>&#8810; conventional</td><td>6% at N=10k</td><td><span className="status_badge validated">Validated</span></td></tr>
                    <tr><td>Penultimate Detection</td><td>100% success</td><td>100%</td><td><span className="status_badge validated">Validated</span></td></tr>
                    <tr><td>Completion O(1)</td><td>Single morphism</td><td>1.0 &#177; 0.0</td><td><span className="status_badge validated">Validated</span></td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      {selectedFig && (
        <Modal isOpen={isOpen} onRequestClose={() => setIsOpen(false)} className="mymodal" overlayClassName="myoverlay" closeTimeoutMS={300}>
          <div className="cavani_tm_modalbox opened">
            <div className="box_inner">
              <div className="close" onClick={() => setIsOpen(false)}>
                <a href="#"><i className="icon-cancel"></i></a>
              </div>
              <div className="description_wrap">
                <div className="service_popup_informations">
                  <div style={{ width: "100%", textAlign: "center", padding: "20px", background: "#0a0f1a", borderRadius: "6px" }}>
                    <img src={selectedFig.img} alt={selectedFig.title} style={{ maxWidth: "100%", maxHeight: "500px", borderRadius: "4px" }} />
                  </div>
                  <div className="details" style={{ marginTop: "20px" }}>
                    <h3 style={{ color: "#e2e8f0" }}>{selectedFig.title}</h3>
                    <span style={{ color: "#0ea5e9" }}>{selectedFig.category}</span>
                  </div>
                  <div className="descriptions" style={{ marginTop: "15px" }}>
                    <p style={{ color: "#a0aab4" }}>{selectedFig.description}</p>
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
