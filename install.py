# ====================================================================================================
# [RT] RareTools Auto-Installer & Dynamic Dependency Resolver
# Resolves and installs requirements and pre-compiled llama-cpp-python CUDA/Metal wheels.
# Target Version: v0.4.1 (JamePeng releases)
# 100% English ASCII Only
# ====================================================================================================

import os
import sys
import platform
import subprocess
import json
import urllib.request
import re

# Releases repository for pre-compiled CUDA wheels
RELEASES_API = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases"
FALLBACK_VERSION = "0.4.1"
RELEASE_DATE = "20260926"

def log(msg):
    print(f"[RT-Installer] {msg}")

def get_platform_info():
    """Detects Python version, OS platform, architecture, and CUDA version."""
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    # OS & Architecture
    os_name = sys.platform
    if os_name.startswith("win"):
        os_tag = "win"
        arch_tag = "win_amd64"
    elif os_name.startswith("linux"):
        os_tag = "linux"
        arch_tag = "linux_x86_64"
    elif os_name.startswith("darwin"):
        os_tag = "macos"
        arch_tag = "macosx_11_0_arm64" if platform.machine() == "arm64" else "macosx"
    else:
        os_tag = os_name
        arch_tag = platform.machine()
        
    # CUDA detection from PyTorch first
    cuda_tag = None
    try:
        import torch
        if torch.cuda.is_available() and torch.version.cuda:
            c_ver = torch.version.cuda
            c_clean = c_ver.replace(".", "")
            if len(c_clean) == 2:
                c_clean += "0"
            cuda_tag = f"cu{c_clean}"
    except Exception:
        pass
        
    # Fallback to nvcc if PyTorch CUDA not detected
    if not cuda_tag and os_tag != "macos":
        try:
            out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
            m = re.search(r"release (\d+\.\d+)", out)
            if m:
                c_clean = m.group(1).replace(".", "")
                if len(c_clean) == 2:
                    c_clean += "0"
                cuda_tag = f"cu{c_clean}"
        except Exception:
            pass

    return py_ver, os_tag, arch_tag, cuda_tag

def fetch_releases():
    """Fetches release list from GitHub API with safety timeout and user-agent."""
    try:
        req = urllib.request.Request(
            f"{RELEASES_API}?per_page=30",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log(f"GitHub API query failed ({e}). Using offline wheel resolver...")
        return None

def find_best_wheel(py_ver, os_tag, arch_tag, cuda_tag, releases=None):
    """Finds the optimal matching wheel from GitHub releases or fallback URLs."""
    if releases:
        matching_assets = []
        for r in releases:
            tag = r.get("tag_name", "")
            
            # Prioritize target version (v0.4.1)
            if f"v{FALLBACK_VERSION}" not in tag and FALLBACK_VERSION not in tag:
                continue

            # Filter by OS
            if os_tag == "macos":
                if "macos" not in tag.lower() and "metal" not in tag.lower():
                    continue
            elif os_tag == "win":
                if "win" not in tag.lower():
                    continue
            elif os_tag == "linux":
                if "linux" not in tag.lower():
                    continue
                    
            for a in r.get("assets", []):
                name = a.get("name", "")
                # Must match Python version and architecture
                if py_ver in name and (arch_tag in name or (os_tag == "macos" and "arm64" in name)):
                    c_match = re.search(r"cu(\d+)", name) or re.search(r"cu(\d+)", tag)
                    asset_cuda = f"cu{c_match.group(1)}" if c_match else ("metal" if "metal" in tag.lower() else "cpu")
                    matching_assets.append({
                        "release": tag,
                        "name": name,
                        "url": a.get("browser_download_url"),
                        "cuda": asset_cuda
                    })
                    
        if matching_assets:
            if os_tag == "macos":
                return matching_assets[0]
                
            if cuda_tag and cuda_tag.startswith("cu"):
                try:
                    user_c_num = int(cuda_tag.replace("cu", ""))
                    def score(item):
                        if item["cuda"].startswith("cu"):
                            item_c_num = int(item["cuda"].replace("cu", ""))
                            penalty = 0 if item_c_num <= user_c_num else 10
                            major_diff = abs(user_c_num // 10 - item_c_num // 10)
                            diff = abs(user_c_num - item_c_num)
                            return (penalty, major_diff, diff)
                        return (999, 999, 999)
                    matching_assets.sort(key=score)
                    return matching_assets[0]
                except Exception:
                    pass
            return matching_assets[0]

    # Offline / Static Fallback URL construction (v0.4.1 JamePeng builds)
    cuda_target = cuda_tag if cuda_tag in ["cu124", "cu126", "cu128", "cu130", "cu131"] else "cu131"
    if os_tag == "win":
        wheel_name = f"llama_cpp_python-{FALLBACK_VERSION}+{cuda_target}-{py_ver}-{py_ver}-win_amd64.whl"
        url = f"https://github.com/JamePeng/llama-cpp-python/releases/download/v{FALLBACK_VERSION}-{cuda_target}-win-{RELEASE_DATE}/{wheel_name.replace('+', '%2B')}"
    elif os_tag == "linux":
        wheel_name = f"llama_cpp_python-{FALLBACK_VERSION}+{cuda_target}-{py_ver}-{py_ver}-linux_x86_64.whl"
        url = f"https://github.com/JamePeng/llama-cpp-python/releases/download/v{FALLBACK_VERSION}-{cuda_target}-linux-{RELEASE_DATE}/{wheel_name.replace('+', '%2B')}"
    elif os_tag == "macos":
        wheel_name = f"llama_cpp_python-{FALLBACK_VERSION}-{py_ver}-{py_ver}-macosx_11_0_arm64.whl"
        url = f"https://github.com/JamePeng/llama-cpp-python/releases/download/v{FALLBACK_VERSION}-Metal-macos-{RELEASE_DATE}/{wheel_name}"
    else:
        return None

    return {
        "name": wheel_name,
        "url": url,
        "cuda": cuda_target if os_tag != "macos" else "metal"
    }

def install_requirements():
    """Installs standard packages from requirements.txt."""
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        log(f"Installing base dependencies from {req_file}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        except subprocess.CalledProcessError as e:
            log(f"Warning: Failed to install some base requirements (code {e.returncode}). Continuing...")

def install_llama_cpp(force=False):
    """Detects environment, resolves wheel URL, and installs llama-cpp-python v0.4.1."""
    if not force:
        try:
            import llama_cpp
            cur_ver = getattr(llama_cpp, "__version__", "unknown")
            if cur_ver == FALLBACK_VERSION or cur_ver.startswith(FALLBACK_VERSION):
                log(f"llama-cpp-python is already at target version {cur_ver}.")
                return True
            else:
                log(f"llama-cpp-python version {cur_ver} detected. Upgrading to target version {FALLBACK_VERSION}...")
        except ImportError:
            pass

    log("Detecting system environment for llama-cpp-python...")
    py_ver, os_tag, arch_tag, cuda_tag = get_platform_info()
    log(f"Platform: Python={py_ver}, OS={os_tag}, Arch={arch_tag}, CUDA={cuda_tag or 'None/CPU'}")

    releases = fetch_releases()
    best_wheel = find_best_wheel(py_ver, os_tag, arch_tag, cuda_tag, releases)

    if not best_wheel:
        log("No matching pre-compiled wheel found. Attempting standard pip install...")
        cmd = [sys.executable, "-m", "pip", "install", f"llama-cpp-python=={FALLBACK_VERSION}"]
    else:
        log(f"Matched Wheel: {best_wheel['name']} (CUDA: {best_wheel['cuda']})")
        log(f"Downloading & installing from: {best_wheel['url']}")
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", best_wheel["url"]]

    try:
        subprocess.check_call(cmd)
        log(f"Successfully installed llama-cpp-python v{FALLBACK_VERSION}!")
        return True
    except subprocess.CalledProcessError as e:
        log(f"Installation via pre-compiled wheel failed with error code {e.returncode}")
        # Secondary fallback: try standard pip without wheel
        try:
            log("Attempting fallback pip install...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"llama-cpp-python=={FALLBACK_VERSION}"])
            return True
        except Exception:
            return False

def ensure_llama_cpp():
    """Helper called by nodes on import if llama_cpp is missing or outdated."""
    try:
        import llama_cpp
        cur_ver = getattr(llama_cpp, "__version__", "unknown")
        if cur_ver == FALLBACK_VERSION or cur_ver.startswith(FALLBACK_VERSION):
            return True
    except ImportError:
        pass
    log(f"llama_cpp not found or outdated. Launching auto-installer for v{FALLBACK_VERSION}...")
    return install_llama_cpp()

if __name__ == "__main__":
    log(f"=== RT-RareTools Installer (v{FALLBACK_VERSION}) ===")
    force = any(arg in sys.argv for arg in ["--force", "-f", "--upgrade", "-U"])
    no_reqs = any(arg in sys.argv for arg in ["--no-reqs", "--llama-only"])
    if not no_reqs:
        install_requirements()
    install_llama_cpp(force=force)
    log("=== Installation Complete ===")
