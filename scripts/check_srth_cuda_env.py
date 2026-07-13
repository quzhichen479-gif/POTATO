from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the local PythonProject2 CUDA environment used for SRTH integration."
    )
    parser.add_argument("--yolo-root", required=True, type=Path)
    parser.add_argument("--import-module", default="ultralytics")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    root = args.yolo_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"YOLO26 project root does not exist: {root}")
    sys.path.insert(0, str(root))

    import torch

    status = {
        "python": sys.executable,
        "python_project2": str(root),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        status["device_0"] = torch.cuda.get_device_name(0)
    try:
        module = __import__(args.import_module)
        status["external_import"] = args.import_module
        status["external_version"] = getattr(module, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - depends on local repository
        status["external_import_error"] = repr(exc)

    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for local SRTH training/integration")
    if "external_import_error" in status:
        raise SystemExit("Could not import the local YOLO26 Python module")


if __name__ == "__main__":
    main()
