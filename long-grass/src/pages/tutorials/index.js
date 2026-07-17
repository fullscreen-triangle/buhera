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

// The intended reading order. Slugs not in this list appear alphabetically
// after the ordered ones.
const ORDER = [
  "basic-routines",
  "kwasa-kwasa-routines",
  "purpose-routines",
  "zangalewa-routines",
  "shapeshifter-routines",
  "scope-routines",
];

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

  // Sort by ORDER, then alphabetical for the rest.
  items.sort((a, b) => {
    const ai = ORDER.indexOf(a.slug);
    const bi = ORDER.indexOf(b.slug);
    if (ai === -1 && bi === -1) return a.slug.localeCompare(b.slug);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });

  return { props: { items } };
}

export default function TutorialsIndex({ items }) {
  return (
    <>
      <Head>
        <title>tutorials · buhera</title>
        <meta name="description" content="Hands-on Buhera OS tutorials, REPL-style." />
      </Head>
      <div className="min-h-screen bg-black text-gray-200">
        <div className="max-w-3xl mx-auto px-6 py-10">
          <nav className="mb-8 text-sm">
            <Link href="/" className="text-blue-400 hover:text-blue-300">
              ← back to terminal
            </Link>
          </nav>

          <h1 className="text-4xl font-bold text-white mb-2">Tutorials</h1>
          <p className="text-gray-400 mb-8 leading-relaxed">
            REPL-style walkthroughs. Every cell is a real command you can type
            into the terminal and get a real result. Follow them in order the
            first time — later routines assume you have done the earlier ones.
          </p>

          <ol className="space-y-6">
            {items.map((item, i) => (
              <li key={item.slug} className="border border-gray-800 rounded p-5 hover:border-gray-600 transition">
                <div className="flex items-baseline gap-3">
                  <span className="text-gray-500 text-sm font-mono">{String(i + 1).padStart(2, "0")}</span>
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
        </div>
      </div>
    </>
  );
}
