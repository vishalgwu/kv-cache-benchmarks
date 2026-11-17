# Pin check: record exactly which package versions this Kaggle session has, and make vLLM match TARGET.
# Paste into ONE Kaggle cell and send me the whole output. It changes nothing except (if needed)
# installing vllm==TARGET. TARGET is the version the Step 7 pilot actually ran on (server log: 0.34.0).
import subprocess, sys, json, os

TARGET = "0.34.0"
KEY = ["vllm", "torch", "transformers", "tokenizers", "huggingface_hub", "numpy", "accelerate", "pytest", "psutil",
       "nvidia-ml-py", "pynvml", "xformers", "triton", "flashinfer-python"]


def sh(cmd, timeout=3000):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def versions():
    # fresh interpreter so a just-finished pip install is seen, not the stale in-memory import
    code = ("import json,importlib.metadata as m\n"
            f"names={KEY!r}\nout={{}}\n"
            "for n in names:\n"
            "    try: out[n]=m.version(n)\n"
            "    except m.PackageNotFoundError: out[n]=None\n"
            "print(json.dumps(out))")
    return json.loads(sh(f"{sys.executable} -c \"{code}\"")[1].strip().splitlines()[-1])


print("python:", sys.version.split()[0])
before = versions()
print("== installed before ==");  print(json.dumps(before, indent=1))
if before["vllm"] != TARGET:
    print(f"vllm {before['vllm']} != {TARGET}: installing vllm=={TARGET} (this also moves torch etc.) ...")
    rc, out = sh(f"pip install -q vllm=={TARGET}", timeout=2400)
    print("pip exit code:", rc); print(out[-1200:])
after = versions()
print("== installed after ==");  print(json.dumps(after, indent=1))
print("vllm matches target:", after["vllm"] == TARGET)

rc, freeze = sh("pip freeze")
open("/kaggle/working/pip_freeze.txt", "w").write(freeze)
print("pip freeze lines:", len(freeze.splitlines()), "-> /kaggle/working/pip_freeze.txt")
print("== key lines from pip freeze ==")
for line in freeze.splitlines():
    if line.split("==")[0].lower().replace("_", "-") in {k.replace("_", "-") for k in KEY}:
        print(line)
print("== gpu / driver ==")
print(sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")[1])
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
