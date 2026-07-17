/* /tutorials/[slug] — a single tutorial page.
 *
 * At build time, reads long-grass/tutorials/<slug>.md, parses it via the
 * tiny in-repo markdown parser, and renders it through TutorialRenderer.
 * Also computes previous/next links based on file order + a canonical
 * reading order for the well-known tutorials.
 */

import fs from "fs";
import path from "path";
import Head from "next/head";
import Link from "next/link";
import { parseTutorial, extractMeta } from "@/lib/tutorial-markdown";
import TutorialRenderer from "@/components/TutorialRenderer";

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
  if (!fs.existsSync(file)) {
    return { notFound: true };
  }
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
    const prev = fs.readFileSync(path.join(dir, `${prevSlug}.md`), "utf-8");
    prevTitle = extractMeta(parseTutorial(prev)).title || prevSlug;
  }
  if (nextSlug) {
    const next = fs.readFileSync(path.join(dir, `${nextSlug}.md`), "utf-8");
    nextTitle = extractMeta(parseTutorial(next)).title || nextSlug;
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

          <TutorialRenderer blocks={blocks} />

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
