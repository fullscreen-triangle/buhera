import Head from "next/head";
import React from "react";
import Preloader from "../components/Preloader";
import Footer from "./footer";
import MagicCursor from "./magic-cursor";

export default function Layout({ children }) {
  return (
    <>
      <Head>
        <meta name="description" content="Buhera - A Categorical Operating System Based on Trajectory Completion" />
        <meta name="author" content="Buhera Research" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <meta name="keywords" content="categorical computing, operating system, trajectory completion, ternary partition tree, S-entropy, zero-cost demons" />

        <title>Buhera | Categorical Operating System</title>

        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet" />
      </Head>
      <Preloader />
      <div className="cavani_tm_all_wrap dark">
        {children}
        <Footer />
        <MagicCursor />
      </div>
      <div className="cavani_fn_moving_box"></div>
    </>
  );
}
