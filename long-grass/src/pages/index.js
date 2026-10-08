import dynamic from "next/dynamic";
import { useRouter } from "next/router";

// The landing page: a document you run (components/notebook/Notebook.js).
// `?doc=<name>` opens another one; the blank surface is at /surface.
const Notebook = dynamic(() => import("@/components/notebook/Notebook"), { ssr: false });

export default function Home() {
  const { query, isReady } = useRouter();
  if (!isReady) return null;
  const name = typeof query.doc === "string" && /^[a-z0-9][a-z0-9-]{0,63}$/.test(query.doc) ? query.doc : "today";
  return <Notebook key={name} name={name} />;
}
