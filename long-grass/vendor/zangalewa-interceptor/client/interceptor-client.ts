/**
 * Portable interceptor client.
 *
 * Drop this into any module site. It is the zoom-climb bridge client with one
 * change that matters: the base URL is a parameter rather than a same-origin
 * assumption, because a broker is cross-origin by definition.
 *
 * No dependencies. Works in any browser context.
 */

export interface InterceptorAgent {
  version?: string;
  hostname?: string;
  dsls?: string[];
  providers?: string[];
  online: boolean;
  claimedByOrigin?: string | null;
}

export interface PairedSession {
  sessionId: string;
  browserToken: string;
  agent: InterceptorAgent;
}

export interface InterceptorOptions {
  /** Broker base URL, e.g. "http://127.0.0.1:4319". No trailing slash needed. */
  baseUrl: string;
  /**
   * sessionStorage key. Give each site its own if several run on one origin,
   * or share one deliberately so they reuse a pairing.
   */
  storageKey?: string;
}

export class Interceptor {
  #base: string;
  #key: string;

  constructor(opts: InterceptorOptions) {
    this.#base = opts.baseUrl.replace(/\/+$/, "");
    this.#key = opts.storageKey ?? "zangalewa.interceptor";
  }

  /**
   * The stored pairing, if any.
   *
   * sessionStorage, not localStorage: this token authorises running code on
   * someone's machine, so it should die with the tab rather than linger on a
   * shared computer. A refresh keeps it; closing the tab does not.
   */
  load(): PairedSession | null {
    if (typeof window === "undefined") return null;
    try {
      const raw = window.sessionStorage.getItem(this.#key);
      return raw ? (JSON.parse(raw) as PairedSession) : null;
    } catch {
      return null;
    }
  }

  save(s: PairedSession | null): void {
    if (typeof window === "undefined") return;
    try {
      if (s) window.sessionStorage.setItem(this.#key, JSON.stringify(s));
      else window.sessionStorage.removeItem(this.#key);
    } catch {
      // Private-browsing quota failure. The pairing simply will not survive a
      // refresh; not worth surfacing.
    }
  }

  /** Spend a pair code. Stores the session on success. */
  async claim(pairCode: string): Promise<PairedSession> {
    const res = await fetch(`${this.#base}/claim`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pairCode }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error || `claim failed (${res.status})`);
    const s: PairedSession = {
      sessionId: body.sessionId,
      browserToken: body.browserToken,
      agent: body.agent,
    };
    this.save(s);
    return s;
  }

  /**
   * Is a machine attached, and what.
   *
   * Never throws on "not paired" — that is an ordinary state a panel renders
   * as a grey dot, not an error.
   */
  async status(): Promise<{ paired: boolean; online: boolean; agent: InterceptorAgent | null }> {
    const s = this.load();
    if (!s) return { paired: false, online: false, agent: null };
    try {
      const res = await fetch(
        `${this.#base}/status?sessionId=${encodeURIComponent(s.sessionId)}`,
        { headers: { Authorization: `Bearer ${s.browserToken}` } }
      );
      if (!res.ok) return { paired: false, online: false, agent: null };
      const b = await res.json();
      return {
        paired: Boolean(b.paired),
        online: Boolean(b.online),
        agent: b.agent ?? null,
      };
    } catch {
      // Broker unreachable is indistinguishable from unpaired, for UI purposes.
      return { paired: false, online: false, agent: null };
    }
  }

  unpair(): void {
    this.save(null);
  }

  /**
   * Run a generation on the paired machine.
   *
   * The 202 path is invisible to the caller: one promise either way. A cold
   * CPU-only model ALWAYS takes it, and prompt evaluation alone can run for
   * minutes, so the default deadline matches the CLI's own per-job budget.
   * A shorter wait would report a timeout while the machine is still working.
   */
  async run(
    payload: unknown,
    opts: { pollMs?: number; maxWaitMs?: number; onPending?: (jobId: string) => void } = {}
  ): Promise<unknown> {
    const s = this.load();
    if (!s) throw new Error("not paired — claim a pair code first");

    const res = await fetch(`${this.#base}/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${s.browserToken}`,
      },
      body: JSON.stringify({ sessionId: s.sessionId, payload }),
    });

    const body = await res.json().catch(() => ({}));

    if (res.status === 409) throw new Error(body?.error || "the CLI is not connected");
    if (res.status === 429) throw new Error(body?.error || "too many jobs in flight");
    if (res.ok && body?.ok) return body.result;
    if (res.ok && body?.error) throw new Error(body.error);

    if (res.status !== 202 || !body?.jobId) {
      throw new Error(body?.error || `interceptor failed (${res.status})`);
    }

    opts.onPending?.(body.jobId);

    const pollMs = opts.pollMs ?? 1500;
    const deadline = Date.now() + (opts.maxWaitMs ?? 11 * 60 * 1000);

    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, pollMs));
      const jr = await fetch(
        `${this.#base}/job?sessionId=${encodeURIComponent(
          s.sessionId
        )}&jobId=${encodeURIComponent(body.jobId)}`,
        { headers: { Authorization: `Bearer ${s.browserToken}` } }
      );
      if (!jr.ok) continue; // transient; keep waiting
      const jb = await jr.json();
      if (jb.status === "done") return jb.result;
      if (jb.status === "failed") throw new Error(jb.error || "generation failed");
      if (!jb.agentOnline) throw new Error("the CLI disconnected mid-job");
    }

    throw new Error("timed out waiting for the CLI");
  }
}
