# Buhera gateway — deployment

Live at **https://buhera-91-98-157-147.sslip.io** on server-2 (`91.98.157.147`).

This records what is actually installed, verified against the running machine
rather than described in advance.

## Layout

| Thing | Where |
|---|---|
| Source | `/srv/buhera` |
| Binary | `/srv/buhera/target/release/buhera-gateway` |
| Database | `/var/lib/buhera/gateway.db` (mode 700, `buhera:buhera`) |
| Signing key | `/etc/buhera/gateway.env` (mode 640, `root:buhera`) |
| Unit | `/etc/systemd/system/buhera-gateway.service` |
| Proxy | `/etc/caddy/Caddyfile` (backup: `Caddyfile.bak-buhera`) |

Binds `127.0.0.1:8090`. Caddy terminates TLS and is the only public path in.

## Why sslip.io

The box has no domain. `<dashed-ip>.sslip.io` resolves to the IP, which is
enough for Let's Encrypt to issue a publicly-trusted certificate. A bare IP
cannot hold one — a self-signed cert on an IP has an empty subject, which
Chrome hard-fails with no way to proceed. Point a real domain here when there
is one; the Caddy block is two lines.

## The signing key

Generated on the host, readable only by root and the service user, never
committed and never transmitted. **Replacing it invalidates every token in
circulation** — sessions and machine pairings alike. That is the blunt
revocation lever; there is not yet a finer one.

```bash
# rotate (logs everyone out, unpairs every machine)
KEY=$(/srv/buhera/target/release/buhera-gateway --db /nonexistent 2>&1 | sed -n '4p' | tr -d ' ')
printf 'BUHERA_GATEWAY_KEY=%s\n' "$KEY" > /etc/buhera/gateway.env
systemctl restart buhera-gateway
```

## Backups — not yet running

The database is the only irreplaceable state: accounts cannot be reconstructed
by replaying anything, and password hashes least of all. The Hetzner project is
administered by a third party whose API token can rebuild this machine
regardless of who holds root, so a copy that lives only on the box is not a
backup.

```bash
# on the box — safe against a concurrently-running service, unlike cp
sqlite3 /var/lib/buhera/gateway.db ".backup '/tmp/gateway-$(date +%F).db'"
```

Pull that off-box on a schedule. **This is not yet automated** — it is the
largest open item in this deployment.

## Update

```bash
# from a workstation
tar czf /tmp/buhera-os.tgz --exclude=target --exclude=node_modules --exclude=.git buhera-os
scp /tmp/buhera-os.tgz olduvai:/tmp/
ssh olduvai 'tar xzf /tmp/buhera-os.tgz -C /srv/buhera --strip-components=1 \
  && cd /srv/buhera && source $HOME/.cargo/env \
  && cargo build --release -p buhera-gateway \
  && systemctl restart buhera-gateway'
```

The database is untouched by an update; the schema is created if absent and
left alone otherwise.

## Accounts

There is no public signup — `/api/auth/signup` was removed. The roster is a
fixed set of profiles, seeded directly against the database:

```bash
# on the box, service can be running — SQLite serialises the write
cat > /tmp/seed.json <<'JSON'
[{"email": "someone@buhera.local", "password": "at least twelve characters"}]
JSON
/srv/buhera/target/release/buhera-gateway --db /var/lib/buhera/gateway.db --seed /tmp/seed.json
shred -u /tmp/seed.json   # or rm -f if shred is unavailable
```

Re-running `--seed` with the same file is safe: an email that already exists
is left untouched (`StoreError::Duplicate`), so the same file can double as
a record of the intended roster without re-hashing existing passwords.

There is currently no way to *change* a seeded password short of dropping the
row from `accounts` and re-seeding — no password-reset flow exists yet.

## Verified on deploy

Checked against the running service, over the public internet:

- Certificate verifies (`ssl_verify_result=0`).
- Login and `/api/run` work end to end; `/api/auth/signup` correctly 404s.
- Content stored in one request is retrievable in the next.
- An account created before a service restart still logs in afterwards.
- Weak passwords and duplicate addresses are refused.
- Wrong password and unknown account are indistinguishable.
- Forged and absent tokens are refused with a bare 401.
- A catalyst token is refused on a session route.
- One account cannot see another's machines.
- The service restarts cleanly and is enabled at boot.

## The relay

`buhera-pair run` (on the paired machine) dials `GET /api/catalysts/relay`
over WebSocket and holds the connection open — the direction is forced,
since a home or office machine sits behind NAT and the gateway can never
reach it directly. `/api/run` dispatches to that connection when the router
picks a live catalyst, executing against the machine's own local
`buhera_kernel::Kernel` rather than the gateway's degraded-path one. Verified
locally end to end: dispatch reaches the machine, state persists across
requests on its kernel, a clean disconnect is detected and reported (not
silently degraded), and `prefer="gateway"` still forces the degraded path.
Not yet verified over the public internet against the live deployment.

Heartbeat interval is 15s from the client; the gateway's liveness window
(`router::LIVENESS_WINDOW_SECS`) is 60s, tolerating three consecutive misses.
A dispatched run that gets no answer within 30s
(`relay::RUN_TIMEOUT`) fails with a clear error rather than hanging the HTTP
request indefinitely.

## Installing `buhera-pair`

Packaged with `cargo-dist` (`[workspace.metadata.dist]` in the workspace
root's `Cargo.toml`), scoped to `buhera-pair` only via `dist = false` on
`buhera-os` and `buhera-gateway` themselves — neither is meant for an end
user to download; `buhera-os`'s demo/repl are developer tools and
`buhera-gateway` is deployed by hand (above). `dist init`/`dist generate`
have no native support for a Cargo workspace that lives in a subdirectory of
the git repo rather than at its root — the case here, since `buhera-os/` sits
inside a repo whose root holds an unrelated legacy `buhera` crate — so the
generated `../../.github/workflows/release.yml` is hand-patched to `cd
buhera-os` before every `dist` invocation. **Re-apply those patches after
any `dist init`/`dist generate` regenerates that file**; `allow-dirty` in
Cargo.toml is what lets `dist` run at all against a workflow that no longer
matches its own template.

Verified locally: `dist build --artifacts=local --target
x86_64-pc-windows-msvc` produces a working `buhera-pair.exe` that pairs and
runs the relay correctly. Not yet verified on GitHub's own runners — that
requires actually pushing a tag and watching the workflow run, which no
local check can substitute for.

Known gotcha: `dist build` compiles the whole workspace under the `[profile.dist]`
profile (not just the released package), and `buhera-os`'s `demo`/`repl`
binaries currently fail to link under it — `libort_sys` (the vendored ONNX
runtime `fastembed` pulls in) is missing several Windows CRT symbols
specifically under `dist`'s settings, though it links fine under plain
`release`. This does not block the `buhera-pair` release (its own artifact
still gets built and packaged despite the unrelated failure elsewhere in the
workspace), but it does mean `dist build`'s exit code and log look alarming
at first glance. Untouched by this work — pre-existing, and out of scope
here since it's `buhera-os`'s own build config, not `buhera-pair`'s.

## What is not here yet

- **Automated backups** (above).
- **Rate limiting** on `/api/auth/*`. Argon2 makes each attempt expensive, but
  nothing yet caps attempts per address.
- **Token revocation** finer than rotating the signing key.
- **A verified CI release.** The workflow is patched and `dist plan`/`dist
  build --artifacts=local` both succeed locally; nobody has yet pushed a tag
  and watched it run for real on GitHub's runners.
- **Code signing.** The Windows and macOS installers this produces are
  unsigned — expect SmartScreen/Gatekeeper warnings until that's addressed.
- **Auto-start on boot / background service registration** for
  `buhera-pair run`. It runs in the foreground today; a user has to keep a
  terminal open (or manage their own service wrapper) to stay dispatchable.
