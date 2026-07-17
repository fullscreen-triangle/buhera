/* ============================================================================
 * TutorialRenderer
 *
 * Renders a block array (from tutorial-markdown.js parseTutorial) as React.
 * Pure — takes blocks in, produces JSX out. No hooks, no state. Callers pass
 * blocks pre-parsed from getStaticProps.
 * ========================================================================== */

import Link from "next/link";

function Inline({ tokens }) {
  return (
    <>
      {tokens.map((t, i) => {
        if (t.kind === "text") return <span key={i}>{t.text}</span>;
        if (t.kind === "code") return (
          <code key={i} className="px-1 py-0.5 rounded bg-gray-900 text-green-300 text-sm font-mono">
            {t.text}
          </code>
        );
        if (t.kind === "bold") return <strong key={i} className="text-white">{t.text}</strong>;
        if (t.kind === "link") {
          const isInternal = t.href.startsWith("./") || t.href.startsWith("/");
          const href = isInternal
            ? (t.href.startsWith("./")
                ? "/tutorials/" + t.href.slice(2).replace(/\.md$/, "")
                : t.href)
            : t.href;
          if (isInternal) {
            return (
              <Link key={i} href={href} className="text-blue-400 hover:text-blue-300 underline">
                {t.text}
              </Link>
            );
          }
          return (
            <a
              key={i}
              href={t.href}
              target="_blank"
              rel="noreferrer"
              className="text-blue-400 hover:text-blue-300 underline"
            >
              {t.text}
            </a>
          );
        }
        return null;
      })}
    </>
  );
}

function CodeBlock({ code, lang }) {
  return (
    <pre className="bg-gray-900 border border-gray-800 rounded p-3 my-3 overflow-x-auto">
      <code className={`text-sm font-mono text-gray-100 language-${lang || "text"}`}>
        {code}
      </code>
    </pre>
  );
}

export default function TutorialRenderer({ blocks }) {
  return (
    <article className="prose prose-invert max-w-none">
      {blocks.map((b, i) => {
        if (b.type === "h1") {
          return <h1 key={i} className="text-3xl font-bold text-white mt-6 mb-4">{b.text}</h1>;
        }
        if (b.type === "h2") {
          return <h2 key={i} className="text-2xl font-semibold text-white mt-6 mb-3 border-b border-gray-800 pb-1">{b.text}</h2>;
        }
        if (b.type === "h3") {
          return <h3 key={i} className="text-xl font-semibold text-gray-100 mt-5 mb-2">{b.text}</h3>;
        }
        if (b.type === "h4") {
          return <h4 key={i} className="text-lg font-medium text-gray-200 mt-4 mb-2">{b.text}</h4>;
        }
        if (b.type === "hr") {
          return <hr key={i} className="my-6 border-gray-800" />;
        }
        if (b.type === "p") {
          return (
            <p key={i} className="text-gray-300 my-3 leading-relaxed">
              <Inline tokens={b.inline} />
            </p>
          );
        }
        if (b.type === "code") {
          return <CodeBlock key={i} code={b.code} lang={b.lang} />;
        }
        if (b.type === "ul") {
          return (
            <ul key={i} className="list-disc pl-6 my-3 text-gray-300 space-y-1">
              {b.items.map((item, j) => (
                <li key={j} className="leading-relaxed">
                  <Inline tokens={item} />
                </li>
              ))}
            </ul>
          );
        }
        if (b.type === "blockquote") {
          return (
            <blockquote key={i} className="border-l-4 border-gray-700 pl-4 my-3 text-gray-400 italic">
              <Inline tokens={b.inline} />
            </blockquote>
          );
        }
        if (b.type === "table") {
          return (
            <div key={i} className="overflow-x-auto my-4">
              <table className="w-full text-sm text-left border border-gray-800">
                <thead className="bg-gray-900 text-gray-200">
                  <tr>
                    {b.header.map((cell, k) => (
                      <th key={k} className="px-3 py-2 border-b border-gray-800">
                        <Inline tokens={cell} />
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {b.rows.map((row, r) => (
                    <tr key={r} className="border-b border-gray-800 text-gray-300">
                      {row.map((cell, c) => (
                        <td key={c} className="px-3 py-2 align-top">
                          <Inline tokens={cell} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return null;
      })}
    </article>
  );
}
