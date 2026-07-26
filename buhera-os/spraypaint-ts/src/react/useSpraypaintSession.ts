// ─────────────────────────────────────────────────────────────────────────────
// useSpraypaintSession — React binding over the headless SpraypaintSession.
//
// The session is the single source of truth for the loop (query, result, stale,
// history). This hook makes React re-render whenever any of that advances, and
// exposes the loop verbs (run, applyGesture, undo, redo) as stable callbacks.
//
// A UI never talks to the binary or mutates a query directly — it calls these
// verbs, and every chart draws from `state`. That is what keeps all three input
// surfaces (prompt, editor, chart gesture) writing to the one AskQuery.
// ─────────────────────────────────────────────────────────────────────────────

import { useCallback, useMemo, useRef, useState } from "react";
import type { AskQuery, AskResult } from "../types.js";
import { DEFAULT_QUERY } from "../types.js";
import type { SpraypaintClient } from "../client.js";
import { SpraypaintSession, type SessionState } from "../session.js";
import type { QueryDiff, QuerySource } from "../crossfilter.js";

export interface UseSpraypaintSession {
  /** Current query, last result, and staleness — everything charts read. */
  state: SessionState;
  /** True while an ask is in flight. */
  running: boolean;
  /** Last error from a run, if any. */
  error: string | null;
  /** Execute the current query against the real binary (increments count). */
  run: () => Promise<void>;
  /** Apply a QueryDiff from any surface; marks stale, records history. */
  applyGesture: (diff: QueryDiff, source: QuerySource) => void;
  /** Apply a gesture AND immediately run — the common "drag then see" path. */
  applyAndRun: (diff: QueryDiff, source: QuerySource) => Promise<void>;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
}

export function useSpraypaintSession(
  client: SpraypaintClient,
  initial: AskQuery = DEFAULT_QUERY,
): UseSpraypaintSession {
  // The session instance is stable across renders.
  const sessionRef = useRef<SpraypaintSession | null>(null);
  if (sessionRef.current === null) {
    sessionRef.current = new SpraypaintSession(client, initial);
  }
  const session = sessionRef.current;

  // A monotonically increasing tick forces re-render when session state advances.
  const [, setTick] = useState(0);
  const bump = useCallback(() => setTick((t) => t + 1), []);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      await session.run();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      bump();
    }
  }, [session, bump]);

  const applyGesture = useCallback(
    (diff: QueryDiff, source: QuerySource) => {
      session.applyGesture(diff, source);
      bump();
    },
    [session, bump],
  );

  const applyAndRun = useCallback(
    async (diff: QueryDiff, source: QuerySource) => {
      session.applyGesture(diff, source);
      bump();
      await run();
    },
    [session, run, bump],
  );

  const undo = useCallback(() => {
    session.undo();
    bump();
  }, [session, bump]);

  const redo = useCallback(() => {
    session.redo();
    bump();
  }, [session, bump]);

  const state = session.state();

  return useMemo<UseSpraypaintSession>(
    () => ({
      state,
      running,
      error,
      run,
      applyGesture,
      applyAndRun,
      undo,
      redo,
      canUndo: session.canUndo(),
      canRedo: session.canRedo(),
    }),
    // state identity changes each bump via session.state(); include running/error.
    [state, running, error, run, applyGesture, applyAndRun, undo, redo, session],
  );
}

export type { AskResult };
