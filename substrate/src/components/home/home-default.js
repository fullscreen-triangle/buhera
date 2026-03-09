import Link from "next/link";
import React from "react";
import { RotateTextAnimation } from "../AnimationText";

export default function HomeDefault({ ActiveIndex, handleOnClick }) {
  return (
    <>
      <div
        className={
          ActiveIndex === 0
            ? "cavani_tm_section active animated fadeInUp"
            : "cavani_tm_section active hidden animated"
        }
        id="home_"
      >
        <div className="cavani_tm_home">
          <div className="content">
            <div className="buhera_logo_hero">
              <img src="img/buhera/logo.png" alt="Buhera" style={{ width: 80, marginBottom: 20, filter: "drop-shadow(0 0 20px rgba(0, 200, 255, 0.3))" }} />
            </div>
            <h3 className="name" style={{ letterSpacing: "6px" }}>BUHERA</h3>
            <p className="tagline" style={{ color: "#8ec8f0", fontSize: "14px", letterSpacing: "3px", marginBottom: "10px", textTransform: "uppercase" }}>
              Categorical Operating System
            </p>
            <span className="line"></span>
            <h3 className="job">
              <RotateTextAnimation />
            </h3>
            <p className="hero_description" style={{ color: "#a0aab4", fontSize: "14px", lineHeight: "1.8", maxWidth: "460px", margin: "20px auto 30px" }}>
              Breaking fundamental complexity barriers through categorical address resolution.
              From O(N log N) to O(log&#8323; N).
            </p>
            <div style={{ display: "flex", gap: "15px", justifyContent: "center", flexWrap: "wrap" }}>
              <div className="cavani_tm_button transition_link">
                <Link href="#framework">
                  <a onClick={() => handleOnClick(1)} style={{ background: "#0ea5e9", borderColor: "#0ea5e9" }}>Explore Framework</a>
                </Link>
              </div>
              <div className="cavani_tm_button transition_link">
                <Link href="#contact">
                  <a onClick={() => handleOnClick(4)}>Get Involved</a>
                </Link>
              </div>
            </div>
            <div className="hero_stats" style={{ display: "flex", gap: "40px", justifyContent: "center", marginTop: "40px", flexWrap: "wrap" }}>
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: "28px", fontWeight: "700", color: "#0ea5e9", display: "block" }}>55x</span>
                <span style={{ fontSize: "11px", color: "#8899a6", textTransform: "uppercase", letterSpacing: "1px" }}>Speedup</span>
              </div>
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: "28px", fontWeight: "700", color: "#10b981", display: "block" }}>R&sup2;=1.0</span>
                <span style={{ fontSize: "11px", color: "#8899a6", textTransform: "uppercase", letterSpacing: "1px" }}>Fit Quality</span>
              </div>
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: "28px", fontWeight: "700", color: "#a78bfa", display: "block" }}>6%</span>
                <span style={{ fontSize: "11px", color: "#8899a6", textTransform: "uppercase", letterSpacing: "1px" }}>Energy Use</span>
              </div>
              <div style={{ textAlign: "center" }}>
                <span style={{ fontSize: "28px", fontWeight: "700", color: "#f59e0b", display: "block" }}>9/9</span>
                <span style={{ fontSize: "11px", color: "#8899a6", textTransform: "uppercase", letterSpacing: "1px" }}>Tests Pass</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
