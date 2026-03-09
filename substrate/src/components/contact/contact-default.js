import React, { useState, useEffect } from 'react'
import { customCursor } from '../../plugin/plugin';

export default function ContactDefault({ ActiveIndex }) {
    useEffect(() => {
        customCursor();
    });

    const [form, setForm] = useState({ email: "", name: "", org: "", msg: "" });
    const [active, setActive] = useState(null);
    const [error, setError] = useState(false);
    const [success, setSuccess] = useState(false);
    const onChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };
    const { email, name, org, msg } = form;
    const onSubmit = (e) => {
        e.preventDefault();
        if (email && name && msg) {
            setSuccess(true);
            setTimeout(() => {
                setForm({ email: "", name: "", org: "", msg: "" });
                setSuccess(false);
            }, 3000);
        } else {
            setError(true);
            setTimeout(() => {
                setError(false);
            }, 3000);
        }
    };

    return (
        <>
            <div className={ActiveIndex === 4 ? "cavani_tm_section active animated fadeInUp" : "cavani_tm_section hidden animated"} id="contact_">
                <div className="section_inner">
                    <div className="cavani_tm_contact">
                        <div className="cavani_tm_title">
                            <span>Get Involved</span>
                        </div>

                        {/* CTA Cards */}
                        <div className="buhera_cta_grid" style={{
                            display: "grid",
                            gridTemplateColumns: "1fr 1fr",
                            gap: "20px",
                            marginBottom: "40px"
                        }}>
                            <div className="buhera_cta_card">
                                <div style={{ fontSize: "28px", marginBottom: "12px" }}>&#x1F52C;</div>
                                <h4>Research Partners</h4>
                                <p>Collaborate on extending the categorical computing framework. We seek partners in theoretical CS, quantum computing, and OS design.</p>
                                <ul className="cta_benefits">
                                    <li>Access to full validation framework</li>
                                    <li>Co-authorship opportunities</li>
                                    <li>Early access to vaHera toolchain</li>
                                </ul>
                            </div>
                            <div className="buhera_cta_card">
                                <div style={{ fontSize: "28px", marginBottom: "12px" }}>&#x1F4C8;</div>
                                <h4>Investors</h4>
                                <p>Categorical computing represents a paradigm shift with applications across scientific computing, cryptography, AI, and databases.</p>
                                <ul className="cta_benefits">
                                    <li>Validated 55x speedup (scaling to 10&#8310;x)</li>
                                    <li>Patent-pending architecture</li>
                                    <li>Multiple market applications</li>
                                </ul>
                            </div>
                        </div>

                        {/* Contact Info */}
                        <div className="short_info">
                            <ul>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-mail-3"></i>
                                        <span><a href="#">research@buhera.dev</a></span>
                                    </div>
                                </li>
                                <li>
                                    <div className="list_inner">
                                        <i className="icon-location"></i>
                                        <span>Open to Global Collaboration</span>
                                    </div>
                                </li>
                            </ul>
                        </div>

                        {/* Contact Form */}
                        <div className="form">
                            <div className="left" style={{ width: "100%" }}>
                                <div className="fields">
                                    <form className="contact_form" onSubmit={(e) => onSubmit(e)}>
                                        <div
                                            className="returnmessage"
                                            style={{ display: success ? "block" : "none" }}
                                        >
                                            <span className="contact_success">
                                                Thank you for your interest. We will be in touch shortly.
                                            </span>
                                        </div>
                                        <div
                                            className="empty_notice"
                                            style={{ display: error ? "block" : "none" }}
                                        >
                                            <span>Please fill in all required fields.</span>
                                        </div>

                                        <div className="fields">
                                            <ul>
                                                <li className={`input_wrapper ${active === "name" || name ? "active" : ""}`}>
                                                    <input
                                                        onFocus={() => setActive("name")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={name}
                                                        name="name"
                                                        type="text"
                                                        placeholder="Your Name *"
                                                    />
                                                </li>
                                                <li className={`input_wrapper ${active === "email" || email ? "active" : ""}`}>
                                                    <input
                                                        onFocus={() => setActive("email")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={email}
                                                        name="email"
                                                        type="email"
                                                        placeholder="Email Address *"
                                                    />
                                                </li>
                                                <li className={`input_wrapper ${active === "org" || org ? "active" : ""}`}>
                                                    <input
                                                        onFocus={() => setActive("org")}
                                                        onBlur={() => setActive(null)}
                                                        onChange={(e) => onChange(e)}
                                                        value={org}
                                                        name="org"
                                                        type="text"
                                                        placeholder="Organization / University"
                                                    />
                                                </li>
                                                <li className={`last ${active === "message" || msg ? "active" : ""}`}>
                                                    <textarea
                                                        onFocus={() => setActive("message")}
                                                        onBlur={() => setActive(null)}
                                                        name="msg"
                                                        onChange={(e) => onChange(e)}
                                                        value={msg}
                                                        placeholder="Tell us about your interest in Buhera *"
                                                    />
                                                </li>
                                            </ul>
                                            <div className="cavani_tm_button">
                                                <input
                                                    className='a'
                                                    type="submit"
                                                    value="Send Inquiry"
                                                    style={{ background: "#0ea5e9", borderColor: "#0ea5e9" }}
                                                />
                                            </div>
                                        </div>
                                    </form>
                                </div>
                            </div>
                        </div>

                        {/* Key Numbers */}
                        <div className="buhera_block" style={{ marginTop: "40px" }}>
                            <h3 className="buhera_subtitle">By The Numbers</h3>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "15px", marginTop: "20px" }}>
                                <div className="buhera_stat_card">
                                    <span className="stat_value" style={{ color: "#0ea5e9" }}>2</span>
                                    <span className="stat_label">Research Papers</span>
                                </div>
                                <div className="buhera_stat_card">
                                    <span className="stat_value" style={{ color: "#10b981" }}>2000+</span>
                                    <span className="stat_label">Lines of Validation Code</span>
                                </div>
                                <div className="buhera_stat_card">
                                    <span className="stat_value" style={{ color: "#a78bfa" }}>5</span>
                                    <span className="stat_label">Publication Figures</span>
                                </div>
                                <div className="buhera_stat_card">
                                    <span className="stat_value" style={{ color: "#f59e0b" }}>7/7</span>
                                    <span className="stat_label">Claims Validated</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
