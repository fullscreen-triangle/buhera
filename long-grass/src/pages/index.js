import dynamic from "next/dynamic";

// The blank surface. Client-only: it reads the page book from localStorage
// and tracks the pointer against the window edges.
const Surface = dynamic(() => import("@/components/surface/Surface"), { ssr: false });

export default function Home() {
  return <Surface />;
}
