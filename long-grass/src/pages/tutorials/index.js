/* /tutorials — the index of Buhera OS tutorials.
 *
 * Reads every .md file in long-grass/tutorials/ at build time, extracts the
 * first h1 as the title and the first paragraph as the description, and
 * renders a table of contents. Each entry links to /tutorials/<slug>.
 */

import fs from "fs";
import path from "path";
import Head from "next/head";
import Link from "next/link";
import { parseTutorial, extractMeta } from "@/lib/tutorial-markdown";
import { START, bySlugOrder } from "@/lib/tutorial-order";


export async function getStaticProps() {
  const dir = path.join(process.cwd(), "tutorials");
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".md"));

  const items = [];
  for (const f of files) {
    const slug = f.replace(/\.md$/, "");
    const raw = fs.readFileSync(path.join(dir, f), "utf-8");
    const blocks = parseTutorial(raw);
    const { title, description } = extractMeta(blocks);
    items.push({ slug, title: title || slug, description });
  }

  items.sort((a, b) => bySlugOrder(a.slug, b.slug));

  return { props: { items } };
}

function List({ items, from }) {
  return (
    <ol className="space-y-6">
      {items.map((item, i) => (
        <li key={item.slug} className="border border-gray-800 rounded p-5 hover:border-gray-600 transition">
          <div className="flex items-baseline gap-3">
            <span className="text-gray-500 text-sm font-mono">{String(i + from).padStart(2, "0")}</span>
            <Link href={`/tutorials/${item.slug}`} className="text-xl font-semibold text-white hover:text-blue-300">
              {item.title}
            </Link>
          </div>
          {item.description && (
            <p className="mt-2 text-gray-400 text-sm leading-relaxed pl-8">
              {item.description}
            </p>
          )}
        </li>
      ))}
    </ol>
  );
}

export default function TutorialsIndex({ items }) {
  return (
    <>
      <Head>
        <title>tutorials · buhera</title>
        <meta name="description" content="Buhera OS tutorials: the blank screen, your mail, finding and planning, jobs on AppHub." />
      </Head>
      <div className="min-h-screen bg-black text-gray-200">
        <div className="max-w-3xl mx-auto px-6 py-10">
          <nav className="mb-8 text-sm">
            <Link href="/" className="text-blue-400 hover:text-blue-300">
              ← back to the blank screen
            </Link>
          </nav>

          <h1 className="text-4xl font-bold text-white mb-2">Tutorials</h1>
          <p className="text-gray-400 mb-6 leading-relaxed">
            Start with the first four: they walk the blank screen and the work
            it is for — your mail, finding things, planning an experiment, and
            running its jobs on AppHub — with one experiment carried through
            all of them. Every cell is something you can type on the blank
            screen; run it here with ▶ and it gives the real result.
          </p>

          <Link
            href="/protein-modelling"
            className="block mb-4 border border-emerald-900/60 rounded p-5 bg-emerald-950/20 hover:border-emerald-700 transition"
          >
            <div className="flex items-baseline gap-3">
              <span aria-hidden className="text-lg">🧬</span>
              <span className="text-xl font-semibold text-emerald-300 hover:text-emerald-200">
                Report — Modelling cytochrome P450 as a knowledge-graph runtime
              </span>
            </div>
            <p className="mt-2 text-gray-400 text-sm leading-relaxed pl-8">
              A full scientific report — introduction, methods, results,
              discussion — ending in a live VSCode-style IDE where you run the
              SBS, shapeshifter, cytochrome, and CKG scripts against the real
              federation.
            </p>
          </Link>

          <List items={items.filter((i) => START.includes(i.slug))} from={1} />

          <h2 className="text-2xl font-semibold text-white mt-12 mb-2">Module guides</h2>
          <p className="text-gray-400 mb-6 leading-relaxed text-sm">
            One module at a time, written for the earlier terminal. Every{" "}
            <code className="text-green-300">dispatch(...)</code> line in them still works typed on the blank screen.
          </p>
          <List items={items.filter((i) => !START.includes(i.slug))} from={START.length + 1} />
        </div>
      </div>
    </>
  );
}
