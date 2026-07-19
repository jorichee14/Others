# Test Forgejo on your own computer (10 minutes)

Run the whole thing on your laptop/desktop first so you know what you're getting
before touching the NAS. This uses a stripped-down setup (SQLite, one container)
that you can delete cleanly afterward.

## 0. Prerequisite: Docker

Install **Docker Desktop** if you don't have it:
- macOS / Windows: <https://www.docker.com/products/docker-desktop/>
- Linux: install `docker` and the `docker compose` plugin from your package manager.

Check it's working:
```bash
docker --version
docker compose version
```

You'll also want **git** and **git-lfs** installed to test cloning:
```bash
git --version
git lfs version   # if missing: https://git-lfs.com/
```

## 1. Start Forgejo

From this `local-test/` folder:
```bash
cd forgejo-nas/local-test
docker compose up -d
docker compose logs -f     # optional: watch it boot, Ctrl-C to stop watching
```

First run pulls the image (~100 MB) — give it a minute.

## 2. Complete the setup wizard

Open **<http://localhost:3000>** in your browser.

You'll see the Forgejo install page. The defaults are fine for a test — just
scroll down and click **Install Forgejo**. (SQLite, the paths, and LFS are
already configured by the compose file.)

> Tip: expand **Administrator Account Settings** at the bottom and set your
> admin username/email/password there. If you skip it, the **first user you
> register** becomes the admin instead.

## 3. Click around — this is the point of the test

- Create a repository (green **+** top-right → **New Repository**).
- Make a second user: your profile menu → **Site Administration → Identity &
  Access → Users → Create User**. Log in as them in a private window to feel the
  permission model.
- Add the second user as a collaborator on your repo (**Repo → Settings →
  Collaborators**) with **Read** vs **Write** and see the difference.

## 4. Test a real clone + push

Grab the HTTPS clone URL from your repo page, then on your machine:
```bash
git clone http://localhost:3000/youradmin/testrepo.git
cd testrepo
echo "hello" > README.md
git add README.md
git commit -m "first commit"
git push        # enter your Forgejo username + password when prompted
```
Refresh the repo page in the browser — your commit should be there.

## 5. Test Git LFS (the whole reason for this)

Inside the same repo:
```bash
git lfs install
git lfs track "*.bin"          # mark .bin files as "large"
git add .gitattributes

# make a fake 50 MB file to stand in for a real big asset
# macOS/Linux:
head -c 50000000 /dev/urandom > big.bin
# (Windows PowerShell: fsutil file createnew big.bin 50000000)

git add big.bin
git commit -m "add big file via LFS"
git push

git lfs ls-files               # should list big.bin -> confirms LFS handled it
```
In the browser, open `big.bin` in the repo — Forgejo shows it as **Stored with
Git LFS**, not as raw binary. That's success.

## 6. Tear it down

When you're done experimenting:
```bash
docker compose down       # stop containers, keep your test data
# ...or wipe everything:
docker compose down
rm -rf data               # deletes all test repos/users
```

Nothing here touches the NAS. When you're happy with how it feels, follow the
main `../README.md` to do the real, HTTPS-protected install on the Synology.

---

### If something doesn't work
- **Port already in use** (`3000` or `2222`): another app has it. Change the
  left-hand number in `docker-compose.yml`, e.g. `"3001:3000"`, and use
  `localhost:3001`.
- **Page won't load:** `docker compose logs server` to see the error.
- **`docker: command not found`:** Docker Desktop isn't running/installed (step 0).
