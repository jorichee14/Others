# Hardware & buying guide — Forgejo Git server for a research lab

For a **small lab (1–10 people)** storing **code + large binary data (rosbags,
point clouds, calibration captures)** with per-user permissions and Git LFS.

## TL;DR

- **Pilot on the Synology you already own — buy nothing yet.** Follow
  `README.md`, invite a labmate, run it for a couple of weeks. Migration to new
  hardware later is a folder copy (see "Migration" below).
- The one thing that actually matters for your data is **storage capacity +
  redundancy + backup**. Compute is a non-issue at this scale.
- Budget priority order: **drives & redundancy → backup → fast network (10GbE)
  → the NAS box itself → UPS.**

---

## The thing to understand before you spend: LFS storage grows fast

Git LFS stores **every version** of every large file. Push a 5 GB rosbag, change
and push it 10 times → that's ~50 GB on the server, not 5. For rosbag/point-cloud
work this adds up quickly, so:

- **Size for 2–3× your expected raw data**, and pick a NAS you can expand.
- **Split your data by how it changes** (strong recommendation for robotics labs):
  - **Git + LFS (Forgejo):** code, configs, calibration *outputs/params*,
    small–medium assets, and datasets you genuinely want versioned.
  - **A plain NAS shared folder (SMB/NFS), or [DVC](https://dvc.org) with the
    NAS as a remote:** giant raw captures that you keep *once* and never really
    "version." Putting terabytes of raw rosbags through LFS works but wastes
    space and slows clones. DVC gives you Git-tracked *pointers* to data living
    on a NAS share/S3 — the usual choice when raw data dwarfs the code.
- Turn on Forgejo's housekeeping: **Site Administration → Maintenance** to garbage-collect
  orphaned LFS objects, and teach the team `git lfs prune` for local cleanup.

---

## What to buy (production)

Since it's grant-funded and reliability matters, do it properly with double
redundancy and a real backup. Concrete shopping list for a small, data-heavy lab:

### 1. The NAS (if you outgrow your current one)
An **8-bay Synology Plus/XS** gives capacity + expandability + room for 10GbE
and an SSD volume:
- **Synology DS1821+** (8-bay, AMD Ryzen, ECC RAM, PCIe slot for 10GbE, 2× NVMe
  slots) — great all-rounder for this use.
- **Synology DS1823xs+** (8-bay, **built-in 10GbE**, ECC) — step up if you want
  10GbE without adding a card.
- **Rackmount (RS-series, e.g. RS2423+)** only if your lab has a server rack.
- A 4–5 bay (DS923+/DS1522+) is fine if data will stay under a few TB, but for
  rosbags you'll likely want the extra bays.

> ⚠️ **Check drive compatibility before buying.** Synology's 2025+ models tightened
> their compatibility list toward Synology-branded drives. Either pick a model
> generation that still allows third-party NAS drives (e.g. the DS1821+ era) or
> budget for Synology-branded drives. Verify on Synology's compatibility page for
> the exact model before purchase.

### 2. Drives — this is where the money should go
- Use **NAS-rated CMR HDDs**: WD Red **Plus**/Pro or Seagate IronWolf/Pro
  (avoid SMR drives — they're bad for RAID).
- **Redundancy: SHR-2 or RAID 6** (survives *two* simultaneous drive failures) —
  worth it for irreplaceable grant/research data.
- **Sizing example:** 8 × 16 TB in SHR-2 ≈ **~96 TB usable**. Start with fewer
  drives and add later — Synology SHR lets you expand by adding/replacing drives.
- Buy drives from **mixed batches / different lots** so they don't all fail together.

### 3. An SSD volume for speed (optional but nice)
Put the **Forgejo database + Git repos on a mirrored SSD volume (NVMe/SATA)** and
**LFS objects on the big HDD RAID**. Result: snappy web UI, browsing, and Git
operations, with bulk data on cheap spinning disks. (You point Forgejo's repo
path and LFS path at different volumes in config.) Alternatively use the NVMe
slots as read/write **cache**.

### 4. Network — matters a lot for rosbags
Gigabit ≈ 110 MB/s; a 5 GB bag takes ~45 s and saturates the link for everyone.
For heavy binary data, get **10GbE** (≈1 GB/s) if the budget allows:
- 10GbE NIC for the NAS (e.g. Synology E10G22-T1-Mini / E10G18-T1) or a model
  with it built in.
- A **10GbE (or 2.5GbE) switch**, and 10GbE/2.5GbE on the workstations that pull
  big data. Even 2.5GbE is a big step up from gigabit if 10GbE is overkill.

### 5. Power protection
A **UPS** (CyberPower/APC, ~1000–1500 VA, USB to the NAS) so a power blip doesn't
corrupt the database mid-write. Synology supports graceful shutdown from UPS.

### 6. Backup — non-negotiable for research data (3-2-1 rule)
Three copies, two media, one offsite:
- **On-NAS RAID is *not* a backup** (it protects against disk failure, not against
  deletion, ransomware, or the NAS dying).
- Add **one of:** a second/older NAS elsewhere on campus (Synology **Snapshot
  Replication** / **Hyper Backup**), or **cloud** (Synology C2, Backblaze B2,
  Wasabi). Back up both `data/` and a Postgres dump (script in `README.md` step 9).

---

## Rough budget tiers (hardware only, USD, approximate)

| Tier | Box | Storage | Extras | Ballpark |
|---|---|---|---|---|
| **Pilot** | Your existing Synology | existing | — | **$0** |
| **Solid** | DS1821+ | 4× 12 TB SHR-2 (~24 TB), add later | UPS | ~$1,800–2,600 |
| **Data-heavy** | DS1821+ / DS1823xs+ | 8× 16–20 TB SHR-2 (~96–120 TB) | NVMe SSD volume, 10GbE NIC + switch, UPS | ~$4,000–7,000 |
| **+ Backup** | any of the above | + second NAS or cloud plan | offsite | +$1,000–2,500 (or cloud/mo) |

Prices move; treat these as planning figures, not quotes.

---

## Migration: pilot → production hardware

Because everything is a folder + a DB, moving is easy and low-risk:

1. On the new NAS, install Container Manager and copy the `forgejo-nas/` folder +
   your filled-in `.env`.
2. Stop Forgejo on the old NAS (`docker compose down`) so data is quiescent.
3. Copy the whole `data/` folder and the `db/` folder (or a `pg_dump`) to the new NAS.
4. `docker compose up -d` on the new NAS. Update `ROOT_URL`/DNS if the address changed.
5. Verify a clone + an LFS pull, then retire the old instance.

No repos are recreated, no history is lost — same data, new box.

---

## Recommendation for your case

You have a Synology and grant money, so:

1. **Now:** pilot on the current NAS (`README.md`) — prove the workflow with the lab.
2. **Decide the data split:** code + calibration outputs in Forgejo/LFS; giant raw
   rosbags either in LFS (if you want them versioned) or on a NAS share/DVC.
3. **Buy:** an 8-bay Synology (DS1821+ class) with SHR-2 NAS drives sized 2–3×
   your data, an SSD volume for repos/DB, 10GbE if you move bags often, a UPS,
   and an offsite/second-copy backup. Verify drive compatibility for the exact
   model first.
