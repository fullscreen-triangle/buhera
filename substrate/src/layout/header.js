import React from 'react'

export default function Header({handleOnClick, ActiveIndex}) {
    return (
        <>
            <div className="cavani_tm_header">
                <div className="logo">
                    <a href="#" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                        <img src="img/buhera/logo.png" alt="Buhera" style={{ height: "32px", filter: "drop-shadow(0 0 8px rgba(14, 165, 233, 0.3))" }} />
                        <span style={{ color: "#e2e8f0", fontSize: "16px", fontWeight: "700", letterSpacing: "4px", fontFamily: "Inter, Poppins, sans-serif" }}>BUHERA</span>
                    </a>
                </div>
                <div className="menu">
                    <ul className="transition_link">
                        <li onClick={() => handleOnClick(0)}><a className={ActiveIndex === 0 ? "active" : ""}>Home</a></li>
                        <li onClick={() => handleOnClick(1)}><a className={ActiveIndex === 1 ? "active" : ""}>Framework</a></li>
                        <li onClick={() => handleOnClick(2)}><a className={ActiveIndex === 2 ? "active" : ""}>Results</a></li>
                        <li onClick={() => handleOnClick(7)}><a className={ActiveIndex === 7 ? "active" : ""}>Components</a></li>
                        <li onClick={() => handleOnClick(3)}><a className={ActiveIndex === 3 ? "active" : ""}>Papers</a></li>
                        <li onClick={() => handleOnClick(4)}><a className={ActiveIndex === 4 ? "active" : ""}>Contact</a></li>
                    </ul>
                </div>
            </div>
        </>
    )
}
