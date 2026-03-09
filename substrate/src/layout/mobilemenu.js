import React, { useEffect } from 'react'
import { dataImage } from '../plugin/plugin'

export default function Mobilemenu({isToggled, handleOnClick}) {
  useEffect(() => {
    dataImage();
  });
    return (
        <>
            <div className={!isToggled ? "cavani_tm_mobile_menu" : "cavani_tm_mobile_menu opened"}>
                <div className="inner">
                    <div className="wrapper">
                        <div className="avatar">
                            <div style={{
                                width: "60px",
                                height: "60px",
                                margin: "0 auto",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center"
                            }}>
                                <img src="img/buhera/logo.png" alt="Buhera" style={{ width: "50px", filter: "drop-shadow(0 0 10px rgba(14, 165, 233, 0.3))" }} />
                            </div>
                        </div>
                        <div className="menu_list">
                            <ul className="transition_link">
                                <li onClick={() => handleOnClick(0)}><a href="#home">Home</a></li>
                                <li onClick={() => handleOnClick(1)}><a href="#framework">Framework</a></li>
                                <li onClick={() => handleOnClick(2)}><a href="#results">Results</a></li>
                                <li onClick={() => handleOnClick(7)}><a href="#components">Components</a></li>
                                <li onClick={() => handleOnClick(3)}><a href="#papers">Papers</a></li>
                                <li onClick={() => handleOnClick(4)}><a href="#contact">Contact</a></li>
                            </ul>
                        </div>
                        <div className="copyright">
                            <p style={{ color: "#64748b" }}>Buhera Research &copy; 2025</p>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
