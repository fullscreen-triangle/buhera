import React from 'react'

export default function AuthorDefault() {
    return (
        <>
            <div className="author_image">
                <div className="main" style={{
                    "background": "linear-gradient(135deg, #0a0f1a 0%, #0d1b2a 30%, #1b2838 60%, #0a0f1a 100%)",
                    "position": "relative",
                    "width": "100%",
                    "height": "100%",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "overflow": "hidden"
                }}>
                    {/* Ternary tree visual background */}
                    <div style={{
                        position: "absolute",
                        top: 0, left: 0, right: 0, bottom: 0,
                        backgroundImage: "url(img/buhera/figure_partition_tree.png)",
                        backgroundSize: "cover",
                        backgroundPosition: "center",
                        opacity: 0.15,
                        filter: "hue-rotate(180deg) saturate(1.5)"
                    }} />
                    {/* Overlay gradient */}
                    <div style={{
                        position: "absolute",
                        top: 0, left: 0, right: 0, bottom: 0,
                        background: "linear-gradient(180deg, rgba(10,15,26,0.3) 0%, rgba(10,15,26,0.7) 50%, rgba(10,15,26,0.95) 100%)"
                    }} />
                    {/* Content */}
                    <div style={{
                        position: "relative",
                        zIndex: 2,
                        textAlign: "center",
                        padding: "40px 30px"
                    }}>
                        <img src="img/buhera/logo.png" alt="Buhera" style={{
                            width: 100,
                            marginBottom: 25,
                            filter: "drop-shadow(0 0 30px rgba(14, 165, 233, 0.4))"
                        }} />
                        <h2 style={{
                            color: "#e2e8f0",
                            fontSize: "26px",
                            fontWeight: "700",
                            letterSpacing: "8px",
                            marginBottom: "8px",
                            fontFamily: "Poppins, sans-serif"
                        }}>BUHERA</h2>
                        <p style={{
                            color: "#64748b",
                            fontSize: "10px",
                            letterSpacing: "4px",
                            textTransform: "uppercase",
                            marginBottom: "30px"
                        }}>Categorical OS Framework</p>
                        <div style={{
                            width: "40px",
                            height: "2px",
                            background: "linear-gradient(90deg, #0ea5e9, #a78bfa)",
                            margin: "0 auto 30px"
                        }} />
                        {/* Key equation */}
                        <div style={{
                            background: "rgba(14, 165, 233, 0.08)",
                            border: "1px solid rgba(14, 165, 233, 0.15)",
                            borderRadius: "8px",
                            padding: "15px 20px",
                            marginBottom: "20px"
                        }}>
                            <p style={{ color: "#0ea5e9", fontSize: "13px", fontFamily: "monospace", letterSpacing: "1px" }}>
                                O(x) = C(x) = P(x)
                            </p>
                            <p style={{ color: "#64748b", fontSize: "9px", marginTop: "5px", letterSpacing: "1px", textTransform: "uppercase" }}>
                                Triple Equivalence
                            </p>
                        </div>
                        {/* Complexity */}
                        <div style={{
                            background: "rgba(167, 139, 250, 0.08)",
                            border: "1px solid rgba(167, 139, 250, 0.15)",
                            borderRadius: "8px",
                            padding: "15px 20px"
                        }}>
                            <p style={{ color: "#a78bfa", fontSize: "13px", fontFamily: "monospace", letterSpacing: "1px" }}>
                                O(N log N) &rarr; O(log&#8323; N)
                            </p>
                            <p style={{ color: "#64748b", fontSize: "9px", marginTop: "5px", letterSpacing: "1px", textTransform: "uppercase" }}>
                                Complexity Reduction
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
