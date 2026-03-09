import React from 'react'

export default function TopBar({toggleTrueFalse, isToggled}) {
    return (
        <>
            <div className="cavani_tm_topbar">
                <div className="topbar_inner">
                    <div className="logo">
                        <a href="#" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <img src="img/buhera/logo.png" alt="Buhera" style={{ height: "28px", filter: "drop-shadow(0 0 8px rgba(14, 165, 233, 0.3))" }} />
                            <span style={{ color: "#e2e8f0", fontSize: "14px", fontWeight: "700", letterSpacing: "3px" }}>BUHERA</span>
                        </a>
                    </div>
                    <div className="trigger">
                        <div onClick={toggleTrueFalse} className={!isToggled ? "hamburger hamburger--slider" : "hamburger hamburger--slider is-active"}>
                            <div className="hamburger-box">
                                <div className="hamburger-inner"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
