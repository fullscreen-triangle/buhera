import React, { useEffect } from "react";
import { dataImage } from "../../plugin/plugin";

export default function AboutDefault({ ActiveIndex }) {
  useEffect(() => {
    dataImage();
  });

  return (
    <>
      <div
        className={
          ActiveIndex === 1
            ? "cavani_tm_section active animated fadeInUp"
            : "cavani_tm_section hidden animated"
        }
        id="about_"
      >
        <div className="section_inner">
          <div className="cavani_tm_about">
            <div className="cavani_tm_title">
              <span>The Framework</span>
            </div>

            {/* Problem Statement */}
            <div className="buhera_block" style={{ marginBottom: "40px" }}>
              <h3 className="buhera_subtitle">The Forward Simulation Problem</h3>
              <p className="buhera_text">
                Modern operating systems execute programs through <strong>forward simulation</strong>: they blindly follow
                instruction sequences without understanding what the computation achieves. A sorting algorithm performs
                O(N log N) comparisons not because the solution requires that complexity, but because the algorithm
                does not <em>know</em> what the sorted state is until it arrives there through exhaustive comparison.
              </p>
              <div className="buhera_problem_grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "15px", marginTop: "20px" }}>
                <div className="buhera_card">
                  <div className="card_icon" style={{ color: "#ef4444", fontSize: "24px", marginBottom: "10px" }}>&#x26A0;</div>
                  <h4>Complexity Barrier</h4>
                  <p>Algorithms cannot break their theoretical lower bounds under forward simulation</p>
                </div>
                <div className="buhera_card">
                  <div className="card_icon" style={{ color: "#f59e0b", fontSize: "24px", marginBottom: "10px" }}>&#x26A1;</div>
                  <h4>Energy Waste</h4>
                  <p>Every computational step dissipates energy, even when fewer operations would suffice</p>
                </div>
                <div className="buhera_card">
                  <div className="card_icon" style={{ color: "#6366f1", fontSize: "24px", marginBottom: "10px" }}>&#x25C9;</div>
                  <h4>Semantic Blindness</h4>
                  <p>Systems cannot leverage knowledge of what they compute to optimize how they compute</p>
                </div>
              </div>
            </div>

            {/* Solution: Trajectory Completion */}
            <div className="buhera_block" style={{ marginBottom: "40px" }}>
              <h3 className="buhera_subtitle">Trajectory Completion</h3>
              <p className="buhera_text">
                Buhera introduces a fundamentally different computational model. Instead of simulating forward from initial
                to final state, the system:
              </p>
              <div className="buhera_steps" style={{ display: "flex", gap: "0", marginTop: "25px", flexWrap: "wrap" }}>
                <div className="buhera_step">
                  <div className="step_num">01</div>
                  <h4>Encode</h4>
                  <p>Desired final state as categorical address in partition space</p>
                </div>
                <div className="buhera_step_arrow">&#8594;</div>
                <div className="buhera_step">
                  <div className="step_num">02</div>
                  <h4>Navigate</h4>
                  <p>To the penultimate state using logarithmic tree traversal</p>
                </div>
                <div className="buhera_step_arrow">&#8594;</div>
                <div className="buhera_step">
                  <div className="step_num">03</div>
                  <h4>Complete</h4>
                  <p>Apply single completion morphism to reach the final state in O(1)</p>
                </div>
              </div>
            </div>

            {/* Triple Equivalence */}
            <div className="buhera_block" style={{ marginBottom: "40px" }}>
              <h3 className="buhera_subtitle">Triple Equivalence Theorem</h3>
              <p className="buhera_text">
                The foundation of Buhera is the discovery that three ostensibly different processes are mathematically identical:
              </p>
              <div className="buhera_equation_box">
                <span className="eq_main">O(x) &equiv; C(x) &equiv; P(x)</span>
                <div className="eq_labels">
                  <span><strong>O(x)</strong> Observation</span>
                  <span><strong>C(x)</strong> Computation</span>
                  <span><strong>P(x)</strong> Partitioning</span>
                </div>
                <p className="eq_note">
                  To observe x is to compute its categorical address is to partition state space around x.
                  If observation and computation are the same operation, then observing the sorted state <em>is</em> the sorting computation.
                </p>
              </div>
            </div>

            {/* Five Core Innovations */}
            <div className="buhera_block" style={{ marginBottom: "40px" }}>
              <h3 className="buhera_subtitle">Five Core Innovations</h3>
              <div className="buhera_innovations_grid">
                <div className="buhera_innovation">
                  <div className="innov_num">01</div>
                  <h4>Categorical Memory Addressing</h4>
                  <p>S-entropy coordinates in ternary partition space replace physical memory locations</p>
                </div>
                <div className="buhera_innovation">
                  <div className="innov_num">02</div>
                  <h4>Penultimate State Scheduling</h4>
                  <p>Processes prioritized by categorical distance to completion rather than time-based quantum allocation</p>
                </div>
                <div className="buhera_innovation">
                  <div className="innov_num">03</div>
                  <h4>Zero-Cost Demon Operations</h4>
                  <p>Exploiting commutation between categorical and physical observables for thermodynamically free operations</p>
                </div>
                <div className="buhera_innovation">
                  <div className="innov_num">04</div>
                  <h4>Proof-Validated Storage</h4>
                  <p>Every memory operation backed by formal verification with zero-knowledge proofs</p>
                </div>
                <div className="buhera_innovation">
                  <div className="innov_num">05</div>
                  <h4>Triple Equivalence Verification</h4>
                  <p>Continuous runtime validation ensuring the fundamental identity holds across all operations</p>
                </div>
              </div>
            </div>

            {/* Architecture Diagram */}
            <div className="buhera_block" style={{ marginBottom: "40px" }}>
              <h3 className="buhera_subtitle">System Architecture</h3>
              <div className="buhera_arch_diagram">
                <div className="arch_layer arch_top">
                  <span className="layer_label">Application Layer</span>
                  <div className="arch_components">
                    <div className="arch_comp">vaHera Programs</div>
                    <div className="arch_comp">System Calls</div>
                    <div className="arch_comp">ZK Proofs</div>
                  </div>
                </div>
                <div className="arch_connector">&#8595;</div>
                <div className="arch_layer arch_mid">
                  <span className="layer_label">Categorical Engine</span>
                  <div className="arch_components">
                    <div className="arch_comp highlight">Trajectory Completion</div>
                    <div className="arch_comp highlight">S-Coordinate Hashing</div>
                    <div className="arch_comp highlight">Penultimate Detection</div>
                  </div>
                </div>
                <div className="arch_connector">&#8595;</div>
                <div className="arch_layer arch_bottom">
                  <span className="layer_label">Foundation</span>
                  <div className="arch_components">
                    <div className="arch_comp">Ternary Partition Tree</div>
                    <div className="arch_comp">Categorical Addressing</div>
                    <div className="arch_comp">Completion Morphisms</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Comparison Table */}
            <div className="buhera_block" style={{ marginBottom: "20px" }}>
              <h3 className="buhera_subtitle">Conventional vs Categorical</h3>
              <div className="buhera_table_wrap">
                <table className="buhera_table">
                  <thead>
                    <tr>
                      <th>Aspect</th>
                      <th>Conventional</th>
                      <th>Buhera</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Paradigm</td>
                      <td>Forward Simulation</td>
                      <td className="highlight_cell">Trajectory Completion</td>
                    </tr>
                    <tr>
                      <td>Sorting</td>
                      <td>O(N log N)</td>
                      <td className="highlight_cell">O(log&#8323; N)</td>
                    </tr>
                    <tr>
                      <td>Speedup at N=10k</td>
                      <td>1x (baseline)</td>
                      <td className="highlight_cell">55x</td>
                    </tr>
                    <tr>
                      <td>Energy</td>
                      <td>100%</td>
                      <td className="highlight_cell">6%</td>
                    </tr>
                    <tr>
                      <td>Demon Cost</td>
                      <td>Landauer minimum</td>
                      <td className="highlight_cell">Zero (commuting)</td>
                    </tr>
                    <tr>
                      <td>Addressing</td>
                      <td>Physical (hex)</td>
                      <td className="highlight_cell">Categorical (S-entropy)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        </div>
      </div>
    </>
  );
}
