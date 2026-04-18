import dynamic from "next/dynamic";

const BuheraTerminal = dynamic(
  () => import("@/components/BuheraTerminal"),
  { ssr: false }
);

export default function Home() {
  return <BuheraTerminal />;
}
