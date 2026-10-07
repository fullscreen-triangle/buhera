/* The tutorials' reading order, in two parts.
 *
 * START: a first task (understanding a specification), then the blank screen
 * and the work it is for — mail, finding things, planning, jobs on AppHub.
 * Read these first, in order.
 *
 * GUIDES: one module at a time, written for the earlier terminal. Every
 * `dispatch(...)` line in them still works typed on the blank screen.
 *
 * Slugs in neither list appear after both, alphabetically.
 */

export const START = [
  "understanding-a-specification",
  "the-blank-screen",
  "your-mail",
  "finding-and-planning",
  "jobs-on-apphub",
];

export const GUIDES = [
  "basic-routines",
  "vahera-dsl",
  "spraypaint-search",
  "interceptor-assistant",
  "vahera-search-catalysts",
  "kwasa-kwasa-routines",
  "purpose-routines",
  "zangalewa-routines",
  "shapeshifter-routines",
  "scope-routines",
  "complete-ckg-experiment",
  "federated-querying",
];

export const ORDER = [...START, ...GUIDES];

export function bySlugOrder(a, b) {
  const ai = ORDER.indexOf(a);
  const bi = ORDER.indexOf(b);
  if (ai === -1 && bi === -1) return a.localeCompare(b);
  if (ai === -1) return 1;
  if (bi === -1) return -1;
  return ai - bi;
}
