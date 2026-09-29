import dynamic from "next/dynamic";

// The legacy scrolling terminal, kept reachable while the blank surface at /
// takes over. Same federation, separate kernel.
const BuheraTerminal = dynamic(() => import("@/components/BuheraTerminal"), { ssr: false });

export default function Terminal() {
  return <BuheraTerminal />;
}
