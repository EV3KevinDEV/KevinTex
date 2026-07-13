"""Write a PyInstaller Windows version resource from a release tag."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def parse_version(value: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(r"v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?", value)
    if not match:
        raise ValueError(f"Version must look like v1, v1.2, or v1.2.3: {value!r}")
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate_version_info.py VERSION")
    version = parse_version(sys.argv[1])
    dotted = ".".join(str(part) for part in version[:3])
    numeric = ", ".join(str(part) for part in version)
    contents = f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({numeric}),
    prodvers=({numeric}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('CompanyName', 'KevinTex'),
          StringStruct('FileDescription', 'KevinTex desktop application'),
          StringStruct('FileVersion', '{dotted}'),
          StringStruct('InternalName', 'KevinTex'),
          StringStruct('OriginalFilename', 'KevinTex.exe'),
          StringStruct('ProductName', 'KevinTex'),
          StringStruct('ProductVersion', '{dotted}')
        ]
      )
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    Path(__file__).with_name("version_info.txt").write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    main()
