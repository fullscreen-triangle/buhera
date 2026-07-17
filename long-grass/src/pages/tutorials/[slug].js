/* /tutorials/[slug] — a single tutorial page.
 *
 * At build time, reads long-grass/tutorials/<slug>.md, parses it via the
 * in-repo markdown parser, computes prev/next links. At mount time,
 * bootstraps the federation and creates a runtime context shared across
 * every RunnableCell on the page — so cells see each other's state
 * (kernel, purpose session, etc.) exactly as in the main terminal.
 */

import fs from "fs";
import path from "path";
import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Head from "next/head";
import Link from "next/link";
import { parseTutorial, extractMeta } from "@/lib/tutorial-markdown";
import { createRuntimeContext } from "@/lib/runtime/run-input";
import { bootstrapFederation } from "@/lib/runtime/bootstrap";

// TutorialRenderer imports RunnableCell → Artifact → the whole render tree.
// Load ssr-off so the tutorial content is SSG-friendly and the heavy render
// tree only lands after hydration.
const TutorialRenderer = dynamic(
  () => import("@/components/TutorialRenderer"),
  { ssr: false }
);

const ORDER = [
  "basic-routines",
  "kwasa-kwasa-routines",
  "purpose-routines",
  "zangalewa-routines",
  "shapeshifter-routines",
  "scope-routines",
];

function orderedSlugs() {
  const dir = path.join(process.cwd(), "tutorials");
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".md"));
  const slugs = files.map((f) => f.replace(/\.md$/, ""));
  slugs.sort((a, b) => {
    const ai = ORDER.indexOf(a);
    const bi = ORDER.indexOf(b);
    if (ai === -1 && bi === -1) return a.localeCompare(b);
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });
  return slugs;
}

export async function getStaticPaths() {
  const slugs = orderedSlugs();
  return {
    paths: slugs.map((slug) => ({ params: { slug } })),
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const dir = path.join(process.cwd(), "tutorials");
  const file = path.join(dir, `${params.slug}.md`);
  if (!fs.existsSync(file)) return { notFound: true };
  const raw = fs.readFileSync(file, "utf-8");
  const blocks = parseTutorial(raw);
  const { title, description } = extractMeta(blocks);

  const slugs = orderedSlugs();
  const idx = slugs.indexOf(params.slug);
  const prevSlug = idx > 0 ? slugs[idx - 1] : null;
  const nextSlug = idx >= 0 && idx < slugs.length - 1 ? slugs[idx + 1] : null;

  let prevTitle = "";
  let nextTitle = "";
  if (prevSlug) {
    prevTitle = extractMeta(parseTutorial(fs.readFileSync(path.join(dir, `${prevSlug}.md`), "utf-8"))).title || prevSlug;
  }
  if (nextSlug) {
    nextTitle = extractMeta(parseTutorial(fs.readFileSync(path.join(dir, `${nextSlug}.md`), "utf-8"))).title || nextSlug;
  }

  return {
    props: {
      slug: params.slug,
      title,
      description,
      blocks,
      prev: prevSlug ? { slug: prevSlug, title: prevTitle } : null,
      next: nextSlug ? { slug: nextSlug, title: nextTitle } : null,
    },
  };
}

export default function TutorialPage({ title, description, blocks, prev, next }) {
  const ctxRef = useRef(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    // Instantiate the runtime context once per page mount.
    ctxRef.current = createRuntimeContext();
    // Register the federation (idempotent — no-op if already registered).
    const cleanup = bootstrapFederation();
    setReady(true);
    return () => { /* keep the federation registered; other pages share it */ };
  }, []);

  return (
    <>
      <Head>
        <title>{title} · tutorials · buhera</title>
        {description && <meta name="description" content={description} />}
      </Head>
      <div className="min-h-screen bg-black text-gray-200">
        <div className="max-w-3xl mx-auto px-6 py-10">
          <nav className="mb-6 text-sm flex items-center justify-between">
            <Link href="/tutorials" className="text-blue-400 hover:text-blue-300">
              ← all tutorials
            </Link>
            <Link href="/" className="text-blue-400 hover:text-blue-300">
              terminal →
            </Link>
          </nav>

          <div className="mb-4 text-xs text-gray-500">
            {ready ? (
              <span>▶ this page is live. code cells run against the real federation.</span>
            ) : (
              <span>booting federation…</span>
            )}
          </div>

          <TutorialRenderer blocks={blocks} ctxRef={ctxRef} />

          <hr className="my-8 border-gray-800" />

          <nav className="flex items-center justify-between text-sm">
            {prev ? (
              <Link
                href={`/tutorials/${prev.slug}`}
                className="text-blue-400 hover:text-blue-300"
              >
                ← {prev.title}
              </Link>
            ) : (
              <span />
            )}
            {next ? (
              <Link
                href={`/tutorials/${next.slug}`}
                className="text-blue-400 hover:text-blue-300"
              >
                {next.title} →
              </Link>
            ) : (
              <span />
            )}
          </nav>
        </div>
      </div>
    </>
  );
}
