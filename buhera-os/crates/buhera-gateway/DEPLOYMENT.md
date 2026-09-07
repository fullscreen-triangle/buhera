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

## Verified on deploy

Checked against the running service, over the public internet:

- Certificate verifies (`ssl_verify_result=0`).
- Signup, login, and `/api/run` work end to end.
- Content stored in one request is retrievable in the next.
- An account created before a service restart still logs in afterwards.
- Weak passwords and duplicate addresses are refused.
- Wrong password and unknown account are indistinguishable.
- Forged and absent tokens are refused with a bare 401.
- A catalyst token is refused on a session route.
- One account cannot see another's machines.
- The service restarts cleanly and is enabled at boot.

## What is not here yet

- **The relay.** `/api/run` routes correctly and reports where work would go,
  but cannot yet reach a paired machine — it returns 503 saying so rather than
  running the work on the gateway, which would answer against the wrong
  filesystem. Until the relay lands, only the gateway path executes.
- **Automated backups** (above).
- **Rate limiting** on `/api/auth/*`. Argon2 makes each attempt expensive, but
  nothing yet caps attempts per address.
- **Token revocation** finer than rotating the signing key.
