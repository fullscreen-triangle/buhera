// /os — web-based Buhera OS REPL.
//
// Renders only on the client (the component talks to a local HTTP
// server via fetch; nothing to pre-render on the Next.js side).

import dynamic from "next/dynamic";
import Head from "next/head";

const OsRepl = dynamic(() => import("@/components/OsRepl"), { ssr: false });

export default function OsPage() {
  return (
    <>
      <Head>
        <title>Buhera OS · web repl</title>
        <meta
          name="description"
          content="Interactive web REPL for a live Buhera kernel."
        />
      </Head>
      <OsRepl />
    </>
  );
}
