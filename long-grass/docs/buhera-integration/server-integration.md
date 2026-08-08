# Buhera — Server Deployment Handover

**Audience:** the server administrator deploying Buhera onto a VM.
**Model:** full handover. You build and run the system; the author hands over
source, this runbook, and nothing else. **No private keys, passwords, or tokens
are contained in this document — and none should ever be sent to you over chat,
email, or a shared doc.** You generate every secret on the VM itself (§5), and
you control SSH access to your own machine (§2).

> A note on the one thing this document deliberately does **not** contain:
> pre-made SSH keys. A private key that travels through a file or a message is a
> compromised key. The correct flow is the reverse — access to *your* VM is
> granted with *your* keys, generated on the machines that will hold them. §2
> walks through it. If anyone offers to email you a private key, refuse it.

---

## 1. What you are deploying

Buhera is a Rust workspace. The deployable artifact is a single container image
built from the repository's multi-stage [`Dockerfile`](../../../Dockerfile),
orchestrated by [`docker-compose.yml`](../../../docker-compose.yml).

| Property | Value |
|---|---|
| Language / build | Rust (stable), built inside the image — no host toolchain needed |
| Runtime base | `debian:bullseye-slim` |
| Primary process | `/app/buhera --config /app/etc/vpos/vpos.conf` |
| Runs as | non-root user `buhera` (uid/gid created in the image) |
| Persistent state | PostgreSQL volume + Redis volume + `./data`, `./logs` |
| Inbound | one HTTP/API port (default **8080**) — see §4 |
| Outbound | may reach companion services (Purpose / Zangalewa / Interceptor) and any external APIs you configure in `.env` |
| Workload shape | mixed: a long-lived API service **and** batch runs that execute to completion and write reports under `./data` |

### System requirements (starting point, tune after first run)

| Resource | Minimum | Comfortable |
|---|---|---|
| vCPU | 4 | 8 |
| RAM | 8 GB | 16 GB |
| Disk | 40 GB SSD | 100 GB SSD |
| OS | Ubuntu 22.04 LTS (or any Docker-capable Linux) | same |
| Software | Docker Engine 24+ and the Docker Compose v2 plugin | same |

The build image is large (it compiles the full Rust workspace and installs
scientific libraries). Expect the **first** `docker compose build` to take
10–30 min and several GB of layer cache. Subsequent builds are fast.

---

## 2. SSH access — you generate the keys, not the author

This is access to *your* VM, so *you* (or whoever logs in) hold the private
keys. Nothing here is provided by the author.

### 2a. If your VM host gave you password login and you want key-only login

On **your own laptop/workstation** (never on a shared machine):

```bash
# ed25519 is the modern default; the comment is just a human label
ssh-keygen -t ed25519 -a 100 -C "buhera-admin@$(hostname)" -f ~/.ssh/buhera_vm
```

This writes two files:

- `~/.ssh/buhera_vm`      → **private key. Never leaves this machine. Never shared.**
- `~/.ssh/buhera_vm.pub`  → public key, safe to copy anywhere.

Install the **public** half on the VM:

```bash
ssh-copy-id -i ~/.ssh/buhera_vm.pub user@YOUR_VM_IP
# or, manually: append the .pub contents to ~/.ssh/authorized_keys on the VM
```

Then connect:

```bash
ssh -i ~/.ssh/buhera_vm user@YOUR_VM_IP
```

### 2b. Harden the VM's SSH once key login works

Edit `/etc/ssh/sshd_config` on the VM:

```
PasswordAuthentication no
PermitRootLogin no
PubkeyAuthentication yes
```

```bash
sudo systemctl restart ssh
```

> Confirm you can still log in with the key **in a second terminal** before you
> close the session that still has password access — otherwise a typo locks you
> out.

### 2c. A deploy key for pulling the repo (optional)

If the repo is private and you pull it onto the VM over SSH, generate a
**separate** key *on the VM* and register its public half as a read-only deploy
key in the Git host UI:

```bash
# run this ON THE VM
ssh-keygen -t ed25519 -a 100 -C "buhera-deploy@vm" -f ~/.ssh/buhera_deploy
cat ~/.ssh/buhera_deploy.pub    # paste this into the Git host as a deploy key
```

Point Git at it:

```bash
echo 'Host github.com
  IdentityFile ~/.ssh/buhera_deploy
  IdentitiesOnly yes' >> ~/.ssh/config
```

The private `buhera_deploy` key stays on the VM and is never copied off it.

---

## 3. Getting the source onto the VM

Either the author ships you a tarball, or you clone (with the deploy key from
§2c):

```bash
# option A: clone
git clone git@github.com:fullscreen-triangle/buhera.git
cd buhera

# option B: tarball the author sends
mkdir buhera && tar -xzf buhera-handover.tar.gz -C buhera && cd buhera
```

---

## 4. Ports and the network boundary — read before first run

The committed `docker-compose.yml` was written for a developer's laptop: it
**publishes every internal service to the host**, including the database. On a
public VM that is unsafe. Treat the compose file as a starting point and apply
the override in §6, which:

- publishes **only** the app's HTTP port, and only on loopback, and
- keeps Postgres, Redis, Prometheus, Grafana, etc. on the internal Docker
  network where the app reaches them by service name.

Ports the compose file *defines* (for reference):

| Service | Container port | Should it be public? |
|---|---|---|
| buhera API | 8080 | **Yes** — behind a reverse proxy / firewall (§7) |
| buhera aux services | 8081–8089 | No unless a specific one is needed |
| PostgreSQL | 5432 | **No** — internal only |
| Redis | 6379 | **No** — internal only |
| Prometheus | 9090 | No — internal, or proxy behind auth |
| Grafana | 3000 | Optional — proxy behind auth if exposed |
| nginx | 80 / 443 | Yes, if you use the bundled proxy |

---

## 5. Secrets — generate them on the VM, never receive them

The repository's compose file contains **placeholder** credentials
(`buhera_password`, Grafana `admin/admin`). **Do not deploy those values.**
Generate real ones on the VM and keep them in a `.env` file, which is already
`.gitignore`d and must never be committed or sent anywhere.

A helper script is provided:
[`scripts/gen-secrets.sh`](../../../scripts/gen-secrets.sh). Run it once on the
VM:

```bash
cd buhera
bash scripts/gen-secrets.sh          # writes .env with fresh random secrets
chmod 600 .env                        # readable only by you
```

It generates: `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`,
and a `BUHERA_API_TOKEN`. Open `.env` afterward and fill in any outbound
service URLs / API keys the author told you about (companion services, external
APIs). If the author needs to give you an external API key, that is *their*
secret to hand over through a secret manager or an encrypted channel — it still
does not belong in this document.

---

## 6. Production compose override

Create `docker-compose.prod.yml` next to the existing compose file. It reads
secrets from `.env` and closes the network boundary. This does **not** modify
the author's committed file; it layers on top of it.

```yaml
# docker-compose.prod.yml — production overlay. Run alongside docker-compose.yml.
services:
  buhera:
    env_file: [.env]
    ports:
      - "127.0.0.1:8080:8080"     # API on loopback only; expose via §7 proxy
    environment:
      - BUHERA_ENVIRONMENT=production
      - RUST_LOG=warn
    restart: unless-stopped

  postgres:
    env_file: [.env]
    environment:
      - POSTGRES_DB=buhera
      - POSTGRES_USER=buhera
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    ports: []                      # <- drop host publish; internal network only

  redis:
    command: ["redis-server", "--requirepass", "${REDIS_PASSWORD}"]
    ports: []                      # <- internal only

  prometheus:
    ports: []                      # <- internal only

  grafana:
    env_file: [.env]
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
    ports:
      - "127.0.0.1:3000:3000"      # loopback only; proxy if you need it remote
```

> The bundled `Dockerfile`'s `production` stage and some compose services
> reference files under a `docker/` directory (nginx, rsyslog, postgres init)
> that are **not** in the repository. Build the **`runtime`** target (which the
> `buhera` service already targets), not `production`, unless the author ships
> that `docker/` directory. Confirm with the author which companion services
> (Purpose / Zangalewa / Interceptor) must actually run — several compose
> services can be omitted for a first deployment.

---

## 7. Bring it up

```bash
cd buhera

# 1. secrets (once)
bash scripts/gen-secrets.sh && chmod 600 .env

# 2. build the image (first build is slow)
docker compose -f docker-compose.yml -f docker-compose.prod.yml build buhera

# 3. start only the core services for a first bring-up
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  up -d buhera postgres redis

# 4. watch it start
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f buhera
```

Stop / restart / update:

```bash
# stop
docker compose -f docker-compose.yml -f docker-compose.prod.yml down

# update to new source, rebuild, restart
git pull        # or extract a new tarball
docker compose -f docker-compose.yml -f docker-compose.prod.yml build buhera
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 8. Exposing the API to the internet

The API is bound to `127.0.0.1:8080` by §6. To serve it publicly, terminate TLS
at a reverse proxy on the host and forward to it. Using Caddy (automatic
Let's Encrypt certificates):

```
# /etc/caddy/Caddyfile
buhera.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

Then restrict the host firewall to only what must be reachable:

```bash
sudo ufw default deny incoming
sudo ufw allow OpenSSH
sudo ufw allow 80,443/tcp
sudo ufw enable
```

Nothing else should be reachable from the internet — Postgres, Redis,
Prometheus, and Grafana all stay on the Docker network or on loopback.

---

## 9. Persistence & backups

State that must survive a container rebuild:

| What | Where |
|---|---|
| PostgreSQL | named volume `postgres-data` |
| Redis | named volume `redis-data` |
| App data / batch reports | `./data` (bind mount) |
| Logs | `./logs` (bind mount) |

Nightly Postgres dump:

```bash
docker compose exec -T postgres pg_dump -U buhera buhera \
  | gzip > backups/buhera-$(date +%F).sql.gz
```

Back up `./data` and the `.env` file (store `.env` encrypted — it holds every
secret) to off-box storage.

---

## 10. Health & first-run verification

```bash
# containers up and healthy?
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps

# app responding on loopback?
curl -fsS http://127.0.0.1:8080/ || echo "no response yet"
```

> The compose healthcheck probes `http://localhost:8080/health`. Confirm with
> the author that this endpoint exists in the running binary; if it does not
> yet, replace the healthcheck test with a TCP probe on 8080, or the container
> will report `unhealthy` even while serving.

---

## 11. What the author still needs to confirm / provide

Hand these questions back to the author before or during deployment — they are
gaps this runbook cannot fill from the source alone:

1. **Companion services.** Which of Purpose / Zangalewa / Interceptor must run,
   and are they separate images, or does the single `buhera` binary reach them
   over the network? Their URLs go in `.env`.
2. **The `docker/` directory.** The `Dockerfile` `production` stage and the
   nginx/postgres-init compose mounts reference files not in the repo. Ship
   them, or confirm the `runtime` target is the intended production build.
3. **Health endpoint.** Does the binary serve `GET /health` on 8080? (§10.)
4. **External API keys.** Any outbound API credentials — delivered through a
   secret manager or encrypted channel, to be pasted into `.env` on the VM.
5. **Domain / TLS.** The hostname to put in the reverse proxy (§8).

---

## 12. Security checklist (before going live)

- [ ] SSH is key-only; password auth and root login disabled (§2b).
- [ ] `.env` generated on the VM, `chmod 600`, never committed or transmitted (§5).
- [ ] No placeholder passwords (`buhera_password`, `admin`) remain anywhere.
- [ ] Postgres and Redis are **not** published to the host (§6).
- [ ] Firewall allows only SSH + 80/443 (§8).
- [ ] TLS terminates at the proxy; the app is not served over plain HTTP publicly.
- [ ] Backups of Postgres + `./data` + encrypted `.env` are running (§9).
- [ ] Container runs as the non-root `buhera` user (already set in the image).
```

