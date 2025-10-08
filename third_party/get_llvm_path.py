import platform
import os


# copied from triton/setup.py::get_llvm_package_info
def get_llvm_path():
    system = platform.system()
    try:
        arch = {"x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine()]
    except KeyError:
        arch = platform.machine()
    if (env_system_suffix := os.environ.get("TRITON_LLVM_SYSTEM_SUFFIX", None)):
        system_suffix = env_system_suffix
    elif system == "Darwin":
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == 'arm64' and is_linux_os('almalinux'):
            system_suffix = 'almalinux-arm64'
        elif arch == 'arm64':
            system_suffix = 'ubuntu-arm64'
        elif arch == 'x64':
            vglibc = tuple(map(int, platform.libc_ver()[1].split('.')))
            vglibc = vglibc[0] * 100 + vglibc[1]
            if vglibc > 228:
                # Ubuntu 24 LTS (v2.39)
                # Ubuntu 22 LTS (v2.35)
                # Ubuntu 20 LTS (v2.31)
                system_suffix = "ubuntu-x64"
            elif vglibc > 217:
                # Manylinux_2.28 (v2.28)
                # AlmaLinux 8 (v2.28)
                system_suffix = "almalinux-x64"
            else:
                # Manylinux_2014 (v2.17)
                # CentOS 7 (v2.17)
                system_suffix = "centos-x64"
        else:
            raise RuntimeError(f"LLVM pre-compiled image is not available for {system}-{arch}.")

    return f"llvm-{system_suffix}"
