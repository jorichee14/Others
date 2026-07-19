# Self-hosted Git server on a Synology NAS — Forgejo + Git LFS

A from-zero guide to running your own private Git server with **user accounts &
permissions** and **Git LFS** (large-file storage) on a Synology NAS, reachable
over the internet with HTTPS.

If you've never done this before, read the "What each piece is" section once,
then follow the numbered steps in order.

---

## What you're building (and why these pieces)

| Piece | What it is | Why |
|---|---|---|
| **Forgejo** | A self-hosted Git service — web UI, user accounts, permissions, issues, and Git over HTTP + SSH. Community fork of Gitea. | The actual "GitHub-at-home." It has **Git LFS built in**, so no extra software. |
| **PostgreSQL** | A database. | Stores users, teams, permissions, issue text, etc. (Your *code and LFS files* live on disk, not in here.) |
| **Docker Compose** | A single `docker-compose.yml` file that defines both containers. | Reproducible: back it up, move it, upgrade it — no clicking to rebuild. On DSM 7.2+ you import it as a **Container Manager → Project**, so you still get the GUI. |
| **Synology Reverse Proxy + Let's Encrypt** | Built into DSM. | Puts **HTTPS** in front of Forgejo so passwords and code aren't sent in the clear. No extra container needed. |

**Git LFS in one sentence:** normal Git stores every version of every file
forever, which is terrible for big binaries (datasets, images, `.psd`, `.zip`,
model weights). LFS swaps those files for tiny text pointers and stores the real
bytes separately — Forgejo hosts both. You'll enable it per-repo with a
`.gitattributes` rule (step 8).

> **Recommendation for you (internet-facing):** do steps 1–5 first on your LAN
> and confirm it works, *then* do steps 6–7 to expose it safely with HTTPS.
> Don't port-forward an HTTP-only server to the internet.

---

## Step 1 — Enable Docker (Container Manager) on the Synology

1. Open **Package Center** in DSM.
2. Install **Container Manager** (called "Docker" on older DSM 6.x).
3. Requires a NAS with an x86/64 CPU and DSM 7.2+ for the Projects feature used
   here. (Most `+` / `x` / `xs` models qualify; entry `j` models with ARM CPUs
   generally don't run Container Manager — check your model if unsure.)

Also enable SSH so you can run a couple of commands:
**Control Panel → Terminal & SNMP → Enable SSH service** (you can turn it off
again afterward).

---

## Step 2 — Create a folder and find your user IDs

1. **Control Panel → Shared Folder → Create** a shared folder, e.g. `docker`.
2. Inside it, create a subfolder `forgejo` (via **File Station**).
3. SSH into the NAS and find the numeric IDs of the user that should own the
   data (your DSM admin user is fine to start):

   ```bash
   ssh your_user@NAS-IP        # e.g. ssh admin@192.168.1.50
   id your_user
   # -> uid=1026(your_user) gid=100(users) ...
   ```

   Note the `uid` and `gid` numbers — they go into `.env` next. Running the
   container as this user avoids file-permission headaches.

---

## Step 3 — Put the config files on the NAS

Copy this whole `forgejo-nas/` folder into the shared folder you made, so you
end up with:

```
/volume1/docker/forgejo/
├── docker-compose.yml
├── .env            <- you create this from .env.example
├── .env.example
└── README.md
```

> `/volume1/...` is the typical path; yours may differ. Find it in File Station
> (right-click the folder → Properties → Location).

Then create your real `.env`:

```bash
cd /volume1/docker/forgejo
cp .env.example .env
vi .env     # or edit it in DSM's Text Editor
```

Fill in:
- `FORGEJO_UID` / `FORGEJO_GID` — the numbers from step 2.
- `POSTGRES_PASSWORD` — a long random password (you won't type it by hand).
- Leave `FORGEJO_DOMAIN` / `FORGEJO_ROOT_URL` as the **NAS IP for now**
  (e.g. `192.168.1.50`). You'll switch to your real domain in step 6.

---

## Step 4 — Start it as a Container Manager Project

**GUI way (recommended):**
1. **Container Manager → Project → Create.**
2. Project name: `forgejo`.
3. Path: browse to `/volume1/docker/forgejo` (where the compose file is).
4. It detects `docker-compose.yml` → **Next → Build**.
5. Watch the log; it pulls the images and starts both containers.

**SSH way (equivalent):**
```bash
cd /volume1/docker/forgejo
sudo docker compose up -d
sudo docker compose logs -f      # Ctrl-C to stop watching
```

Now open **`http://NAS-IP:3000`** in a browser. You should see Forgejo.
(Because we locked the installer, it goes straight to the app — no setup wizard.)

---

## Step 5 — Create your admin account

Because we disabled open registration, create the first admin from the command
line:

```bash
sudo docker exec -it forgejo forgejo admin user create \
  --admin \
  --username youradmin \
  --email you@example.com \
  --password 'a-strong-password' \
  --must-change-password=false
```

Log in at `http://NAS-IP:3000` with that account. You're now the administrator.

---

## Step 6 — Put HTTPS + a domain in front (do this before exposing it)

You need three things: a **name**, a **certificate**, and a **reverse proxy**.

### 6a. Get a hostname
- If you have a domain (e.g. `example.com`), create an A record
  `git.example.com` → your home IP. If your home IP changes, use **DDNS**
  (**Control Panel → External Access → DDNS**; Synology offers free
  `yourname.synology.me` names).
- Forward **port 443** (HTTPS) from your router to the NAS.
- Forward the **Git SSH port 2222** from your router to the NAS too (for
  `git clone ssh://…`).

### 6b. Get a free TLS certificate
**Control Panel → Security → Certificate → Add → Add a new certificate → Get a
certificate from Let's Encrypt.** Domain: `git.example.com`. DSM auto-renews it.

### 6c. Reverse proxy Forgejo behind HTTPS
**Control Panel → Login Portal → Advanced → Reverse Proxy → Create:**
- **Source:** Protocol `HTTPS`, Hostname `git.example.com`, Port `443`.
- **Destination:** Protocol `HTTP`, Hostname `localhost`, Port `3000`.
- On the **Custom Header** tab → **Create → WebSocket** (adds the headers Git's
  web UI needs).

Then assign the Let's Encrypt cert to this hostname under
**Certificate → Settings**.

### 6d. Tell Forgejo its real URL
Edit `.env`:
```
FORGEJO_DOMAIN=git.example.com
FORGEJO_ROOT_URL=https://git.example.com/
```
Re-apply (**Container Manager → Project → forgejo → Action → Build**, or
`sudo docker compose up -d`). Now clone URLs shown in the UI will be correct:
`https://git.example.com/...` and `ssh://git@git.example.com:2222/...`.

---

## Step 7 — Users, teams & permissions

Forgejo's model, briefly:
- **Users** — individual accounts. As admin: **Site Administration (top-right)
  → Identity & Access → Users → Create User**. (Or turn registration back on
  temporarily and let people self-serve — your call.)
- **Organizations** — shared owners for a group of repos (like a company).
- **Teams** inside an org — grant a set of users a permission level across many
  repos at once: **Read**, **Write**, or **Admin**, and you can scope a team to
  specific units (code, issues, wiki, LFS…).
- **Per-repo collaborators** — for one-off access, add a single user to one repo
  with Read/Write/Admin: **Repo → Settings → Collaborators**.
- **Branch protection** — **Repo → Settings → Branches** to require reviews or
  block force-pushes on `main`.

Typical setup: make an **Organization** for your team, create a **Team**
("developers" = Write, "viewers" = Read), add users to teams. New repos in the
org inherit those team permissions automatically.

---

## Step 8 — Using Git LFS

The LFS *server* is already on (we set `LFS_START_SERVER=true`). Each person
who pushes big files installs the LFS client once and marks which file types are
"large":

```bash
# one-time on your computer
git lfs install

# inside a repo — track big file types (creates/edits .gitattributes)
cd my-repo
git lfs track "*.zip"
git lfs track "*.psd"
git lfs track "*.bin"
git add .gitattributes

# from now on, matching files are stored via LFS automatically
git add big_dataset.zip
git commit -m "Add dataset via LFS"
git push
```

Commit the `.gitattributes` file — that's what tells every collaborator (and
Forgejo) which files go through LFS. Verify with `git lfs ls-files`.

Optional guardrails you can set later in `app.ini` (inside `data/gitea/conf/`)
or via env: a max file size, and periodic cleanup of orphaned LFS objects
(**Site Administration → Maintenance**).

---

## Step 9 — Backups (do this before you rely on it)

Everything that matters is in two folders on the NAS: `data/` (repos + LFS +
config) and `db/` (the database). Two good options:

1. **Synology Hyper Backup** the `docker/forgejo` shared folder on a schedule.
   Simple, but back up when the DB is quiescent, or use option 2 for a
   consistent DB dump.
2. **Scripted dump** (cron via **Control Panel → Task Scheduler**):
   ```bash
   # database dump
   sudo docker exec forgejo-db pg_dump -U forgejo forgejo > /volume1/backups/forgejo-db.sql
   # repos + LFS + config
   tar czf /volume1/backups/forgejo-data.tgz -C /volume1/docker/forgejo data
   ```
   Store copies off the NAS too (a NAS is not a backup of itself).

Forgejo also has a built-in dump: `docker exec -u git forgejo forgejo dump`.

---

## Step 10 — Upgrades

Because it's a Project, upgrades are painless:
```bash
cd /volume1/docker/forgejo
sudo docker compose pull      # get the newer image for the pinned major (11)
sudo docker compose up -d     # recreate containers; DB migrates automatically
```
The compose file pins the **major** version (`:11`) so `pull` gets safe patch
updates without a surprise jump. To move to the next major (e.g. 12), **read
that release's notes first**, take a backup, then change `:11` in
`docker-compose.yml` and repeat the two commands.

---

## Security checklist (internet-facing)

- [ ] HTTPS working; `http://…:3000` not exposed to the internet — only 443
      (via reverse proxy) and 2222 (Git SSH) forwarded on the router.
- [ ] `DISABLE_REGISTRATION=true` (set) — you create accounts.
- [ ] Strong admin password; enable **2FA** for admins (Settings → Security).
- [ ] Keep repos **private** by default (set).
- [ ] Synology firewall on (**Control Panel → Security → Firewall**), allowing
      only the ports you need.
- [ ] Regular backups running (step 9) and tested by restoring once.
- [ ] Keep Forgejo and DSM updated.

---

## Quick reference

| Action | Command |
|---|---|
| Start / apply changes | `docker compose up -d` |
| Stop | `docker compose down` |
| Logs | `docker compose logs -f server` |
| Create admin | `docker exec -it forgejo forgejo admin user create --admin …` |
| Reset a password | `docker exec -it forgejo forgejo admin user change-password -u name -p 'new'` |
| DB backup | `docker exec forgejo-db pg_dump -U forgejo forgejo > dump.sql` |
| Upgrade | `docker compose pull && docker compose up -d` |

- Clone over HTTPS: `git clone https://git.example.com/org/repo.git`
- Clone over SSH: `git clone ssh://git@git.example.com:2222/org/repo.git`
  (add your SSH public key in Forgejo: **Settings → SSH / GPG Keys** first).

Docs: <https://forgejo.org/docs/> · Git LFS: <https://git-lfs.com/>
