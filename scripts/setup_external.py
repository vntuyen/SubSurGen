"""Fetch the official baseline repositories (cross-platform).



Usage (from the project root):
    python scripts/setup_external.py
    python scripts/setup_external.py --list          # show status only
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.environ.get("SUBSURGEN_EXTERNAL", os.path.join(ROOT, "external"))

REPOS = {
    "ctabganplus": "Team-TUD/CTAB-GAN-Plus",
}


def have_git() -> bool:
    return shutil.which("git") is not None


def clone_with_git(repo: str, dest: str) -> bool:
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", dest],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:                                        # noqa: BLE001
        return False


def download_tarball(repo: str, dest: str) -> bool:
    """Fallback: fetch the tarball directly, no git required."""
    for branch in ("main", "master"):
        url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                if r.status != 200:
                    continue
                blob = r.read()
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                members = tar.getmembers()
                if not members:
                    continue
                prefix = members[0].name.split("/")[0] + "/"
                os.makedirs(dest, exist_ok=True)
                for m in members:
                    if not m.name.startswith(prefix):
                        continue
                    rel = m.name[len(prefix):]
                    if not rel:
                        continue
                    m.name = rel
                    tar.extract(m, dest)          # nosec - trusted GitHub source
            return True
        except Exception:                                    # noqa: BLE001
            continue
    return False


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch official baseline repositories")
    p.add_argument("--only", nargs="*", default=None, choices=list(REPOS))
    p.add_argument("--list", action="store_true", help="show status and exit")
    p.add_argument("--force", action="store_true", help="re-fetch even if present")
    args = p.parse_args()

    os.makedirs(EXTERNAL, exist_ok=True)
    print(f"external directory: {EXTERNAL}")
    print(f"git available: {have_git()}\n")

    if args.list:
        for folder, repo in REPOS.items():
            present = os.path.isdir(os.path.join(EXTERNAL, folder))
            print(f"  [{'x' if present else ' '}] {folder:14s} {repo}")
        return 0

    targets = args.only or list(REPOS)
    ok, failed = [], []
    for folder in targets:
        repo = REPOS[folder]
        dest = os.path.join(EXTERNAL, folder)
        if os.path.isdir(dest) and not args.force:
            print(f"  [skip] {folder:14s} already present")
            ok.append(folder)
            continue
        if args.force and os.path.isdir(dest):
            shutil.rmtree(dest, ignore_errors=True)

        print(f"  [get ] {folder:14s} {repo} ...", end=" ", flush=True)
        got = (clone_with_git(repo, dest) if have_git()
               else download_tarball(repo, dest))
        if not got:                                   # try the other route
            got = (download_tarball(repo, dest) if have_git()
                   else False)
        if got:
            print("ok")
            ok.append(folder)
        else:
            print("FAILED")
            failed.append((folder, repo))

    print(f"\n{len(ok)}/{len(targets)} repositories available")
    if failed:
        print("\nfailed - clone these manually:")
        for folder, repo in failed:
            print(f"  git clone https://github.com/{repo}.git "
                  f"{os.path.join(EXTERNAL, folder)}")

    print("\npip-installable baselines:")
    print("  pip install sdv                  # GaussianCopula, CTGAN, TVAE")
    print("  pip install synthcity            # TabDDPM, SurVAE, SurvivalGAN, "
          "SurvivalCTGAN")
    print("  # synthcity + torch compatibility issues are usually opacus, not")
    print("  # torch itself -- see subsurgen/baselines.py:_torch_supports_synthcity")
    print("\nEach cloned repo has its own requirements.txt. If its dependencies")
    print("conflict with this project, generate in a separate environment and drop")
    print("the CSVs into results/synthetic/<Method>/<Cohort>.csv - step 3 picks")
    print("them up automatically.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
