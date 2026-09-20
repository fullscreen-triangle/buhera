import "@/styles/globals.css";
import Head from "next/head";
import { useRouter } from "next/router";
import { useEffect } from "react";
import { useGatewaySession } from "@/lib/auth/useGatewaySession";

// Routes reachable without a gateway session. Everything else redirects to
// /login, since the gateway is the only account system this app has.
const PUBLIC_ROUTES = new Set(["/login"]);

function AuthGate({ children }) {
  const router = useRouter();
  const { loggedIn, checked } = useGatewaySession();
  const isPublic = PUBLIC_ROUTES.has(router.pathname);

  useEffect(() => {
    if (!checked || isPublic || loggedIn) return;
    router.replace(`/login?next=${encodeURIComponent(router.asPath)}`);
  }, [checked, isPublic, loggedIn, router]);

  // Before the localStorage check has run (SSR / first paint) or while a
  // redirect to /login is pending, render nothing rather than flashing the
  // terminal — the check is synchronous-fast (localStorage), so this is a
  // single frame, not a spinner-worthy wait.
  if (!isPublic && (!checked || !loggedIn)) return null;

  return children;
}

export default function App({ Component, pageProps }) {
  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>buhera</title>
        <meta name="description" content="a research operating system" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <AuthGate>
        <Component {...pageProps} />
      </AuthGate>
    </>
  );
}
