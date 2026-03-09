import React, { useState } from 'react'
import Modal from 'react-modal';

export default function Service({ ActiveIndex }) {
    const [isOpen, setIsOpen] = useState(false);
    const [modalContent, setModalContent] = useState({});

    function toggleModal() {
        setIsOpen(!isOpen);
    }

    const figures = [
        {
            img: "img/buhera/figure_sorting.png",
            title: "Sorting Complexity",
            subtitle: "O(log\u2083 N) validated with R\u00B2 = 1.0",
            desc: "Categorical sorting achieves logarithmic complexity in base-3, breaking the conventional O(N log N) barrier.",
            detail1: "Linear regression on the operation count yields slope 0.95 with perfect fit quality R\u00B2 = 1.000, confirming the theoretical O(log\u2083 N) prediction. Conventional QuickSort matches expected O(N log N) with R\u00B2 = 0.9999.",
            detail2: "Measured speedups: 3.5x at N=100, 25x at N=1,000, 55x at N=10,000. The trend is strictly increasing, confirming asymptotic advantage. Extrapolation predicts 1,700x at N=10\u2076 and 170,000x at N=10\u2078.",
            detail3: "Energy consumption ratio: categorical uses only 6% of conventional energy at N=10\u2074, decreasing to 2% at N=10\u2075. Each navigation step updates a 64-bit address register rather than performing data comparisons and swaps."
        },
        {
            img: "img/buhera/figure_commutation.png",
            title: "Commutation Relations",
            subtitle: "All 9 tests pass < 10\u207B\u00B9\u2070",
            desc: "Categorical operators commute with physical operators, enabling zero-cost demon operations.",
            detail1: "We construct quantum operators in the hydrogen atom basis |n,l,m,s\u27E9 for principal quantum number n \u2264 5 (Hilbert space dimension 56). Categorical operators (\u0302n, \u0302l, \u0302m) are diagonal; physical operators (\u0302x, \u0302p, \u0302H) have off-diagonal elements.",
            detail2: "All nine commutator magnitudes [O_cat, O_phys] measure below 10\u207B\u00B9\u2070, confirming exact commutation within numerical precision. Residual values are 6-8 orders of magnitude below typical matrix elements.",
            detail3: "Finite size scaling analysis shows commutator residuals vanish as n_max\u207B\u00B2, confirming they arise from finite truncation. In the infinite-dimensional limit, commutators are exactly zero."
        },
        {
            img: "img/buhera/figure_partition_tree.png",
            title: "Ternary Partition Trees",
            subtitle: "37% fewer steps than binary",
            desc: "Ternary trees provide optimal navigation structure for categorical addressing.",
            detail1: "At depth d, the ternary tree has 3\u1D48 leaf nodes, each representing a unique categorical address. Navigation from any node to any other requires O(log\u2083 N) steps, with 37% fewer levels than binary trees: 1 - ln2/ln3 \u2248 0.37.",
            detail2: "For depth 20, ternary capacity is 3.49 \u00D7 10\u2079 addressable locations with 20-digit ternary addresses, compared to 2\u00B2\u2070 = 1.05 \u00D7 10\u2076 for binary trees of the same depth.",
            detail3: "Experimental validation with 1000 random navigation trials confirms measured step counts match theoretical predictions within \u00B11 step across all tested tree sizes."
        },
        {
            img: "img/buhera/figure_s_coordinates.png",
            title: "S-Entropy Coordinates",
            subtitle: "Universal state descriptor",
            desc: "Every system state uniquely specified by S = (S\u2096, S\u209C, S\u2091) in normalized entropy space.",
            detail1: "S\u2096 (kinetic entropy) measures momentum distribution disorder. S\u209C (thermal entropy) measures energy distribution disorder. S\u2091 (exchange entropy) measures particle correlation disorder. Each is normalized to [0,1].",
            detail2: "Address resolution through hierarchical refinement: starting from coarse partition (depth 1), each level reduces error by factor ~3. After 5 levels, reconstruction error < 0.1%. At depth 10, resolution is 3\u207B\u00B9\u2070 \u2248 1.7 \u00D7 10\u207B\u2075.",
            detail3: "Different systems occupy distinct S-space regions: sorted arrays at (0.1, 0.05, 0.2), random data at (0.5, 0.5, 0.5), hot gas at (0.9, 0.95, 0.7). This provides a universal coordinate system for any physical or computational state."
        },
        {
            img: "img/buhera/figure_processor.png",
            title: "Categorical Processor",
            subtitle: "O(log\u2083 N) + O(1) completion",
            desc: "Five-stage pipeline: Input, Resolve, Navigate, Complete, Output.",
            detail1: "The categorical processor operates in 5 stages: (1) Input: receive data and target \u2014 O(1), (2) Resolve address: compute categorical address \u2014 O(log\u2083 N), (3) Navigate: traverse partition tree \u2014 O(log\u2083 N), (4) Complete: apply single morphism \u2014 O(1), (5) Output: return result \u2014 O(1).",
            detail2: "Trajectory completion vs forward simulation: at N=10, speedup is 4.5x. At N=10\u2076, speedup reaches 77,000x. The completion morphism is consistently O(1) \u2014 exactly 1 operation independent of N.",
            detail3: "Penultimate state detection achieves 100% success rate across all tested configurations. Average convergence: 7.2 \u00B1 0.8 steps for N=1000 (theoretical: ceil(log\u2083 1000) = 7)."
        },
        {
            img: "img/buhera/logo.png",
            title: "vaHera Language",
            subtitle: "Compile-time complexity verification",
            desc: "Domain-specific language with dependent types that prove complexity bounds at compile time.",
            detail1: "vaHera enforces categorical correctness through its type system. Types include SCoord (S-entropy coordinates), Addr (ternary addresses), Process (process descriptors with state invariants), and Proof<T> (zero-knowledge proofs).",
            detail2: "The compiler verifies complexity bounds declared in function signatures. If an implementation exceeds its declared bound, compilation fails with a detailed error message suggesting categorical alternatives.",
            detail3: "Built on Rust for memory safety and zero-cost abstractions. The vaHera compiler generates categorical bytecode that executes on the Buhera OS kernel, with full formal verification of correctness properties."
        }
    ];

    return (
        <>
            <div className={ActiveIndex === 7 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="news_">
                <div className="section_inner">
                    <div className="cavani_tm_service">
                        <div className="cavani_tm_title">
                            <span>Core Components</span>
                        </div>
                        <div className="service_list">
                            <ul>
                                {figures.map((item, i) => (
                                    <li key={i}>
                                        <div className="list_inner" onClick={() => { setModalContent(item); toggleModal(); }}>
                                            <div className="buhera_figure_thumb" style={{
                                                width: "100%",
                                                height: "140px",
                                                backgroundImage: `url(${item.img})`,
                                                backgroundSize: "cover",
                                                backgroundPosition: "center",
                                                borderRadius: "6px",
                                                marginBottom: "15px",
                                                border: "1px solid rgba(14, 165, 233, 0.15)"
                                            }} />
                                            <h3 className="title" style={{ color: "#e2e8f0", marginBottom: "4px" }}>{item.title}</h3>
                                            <p style={{ color: "#0ea5e9", fontSize: "12px", marginBottom: "8px" }}>{item.subtitle}</p>
                                            <p className="text" style={{ color: "#8899a6" }}>{item.desc}</p>
                                            <a className="cavani_tm_full_link" href="#" onClick={(e) => { e.preventDefault(); setModalContent(item); toggleModal(); }} />
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            {modalContent && (
                <Modal
                    isOpen={isOpen}
                    onRequestClose={toggleModal}
                    contentLabel="Component Detail"
                    className="mymodal"
                    overlayClassName="myoverlay"
                    closeTimeoutMS={300}
                    openTimeoutMS={300}
                >
                    <div className="cavani_tm_modalbox opened">
                        <div className="box_inner">
                            <div className="close" onClick={toggleModal}>
                                <a href="#"><i className="icon-cancel"></i></a>
                            </div>
                            <div className="description_wrap">
                                <div className="service_popup_informations">
                                    <div className="image">
                                        <img src="img/thumbs/4-2.jpg" alt="" />
                                        <div className="main" style={{
                                            backgroundImage: `url(${modalContent.img})`,
                                            backgroundSize: "contain",
                                            backgroundRepeat: "no-repeat",
                                            backgroundPosition: "center",
                                            backgroundColor: "#0a0f1a"
                                        }} />
                                    </div>
                                    <div className="details" style={{ marginTop: "20px" }}>
                                        <h3 style={{ color: "#e2e8f0" }}>{modalContent.title}</h3>
                                        <span style={{ color: "#0ea5e9" }}>{modalContent.subtitle}</span>
                                    </div>
                                    <div className="descriptions" style={{ marginTop: "15px" }}>
                                        <p style={{ color: "#a0aab4" }}>{modalContent.detail1}</p>
                                        <p style={{ color: "#a0aab4" }}>{modalContent.detail2}</p>
                                        <p style={{ color: "#a0aab4" }}>{modalContent.detail3}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </Modal>
            )}
        </>
    )
}
