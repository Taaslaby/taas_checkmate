#!/usr/bin/env python3
"""Interactive TUI Setup & Installer for taas_checkmate neural engine.

Features:
- Bilingual interface (RU / EN)
- System requirement diagnostics (rustc, maturin, nvcc, python)
- Modular compilation feature selection (connectome, t3, cuda, all)
- Error reporting & automatic dependency resolution
"""

import sys
import os
import subprocess
import shutil

# Check if rich is installed, if not fallback to basic ANSI or auto-install
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
except ImportError:
    print("Installing 'rich' for TUI setup wizard...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint

console = Console()

TEXTS = {
    "RU": {
        "title": "⚡ TAAS Engine — Интерактивный Мастер Установки & Сборки ⚡",
        "select_lang": "Выберите язык / Select language:",
        "sys_check": "🔍 Диагностика системного окружения",
        "tool": "Инструмент",
        "status": "Статус",
        "version": "Версия / Детали",
        "found": "[bold green]Обнаружен[/bold green]",
        "missing": "[bold red]Не найден[/bold red]",
        "optional_missing": "[bold yellow]Опционально (не найден)[/bold yellow]",
        "comp_config": "⚙️ Настройка конфигурации компиляции Rust & CUDA",
        "select_features": "Выберите фичи для сборки:",
        "opt_default": "1. Стандартный (Core + SIMD + Connectome + T3)",
        "opt_cuda": "2. GPU Ускорение (+ CUDA C++ ядра)",
        "opt_minimal": "3. Минимальный (Только Core SIMD)",
        "opt_all": "4. Полный сборник (Все фичи + Python + CUDA)",
        "build_profile": "Выберите профиль сборки:",
        "prof_release": "1. Release (Максимальная оптимизация - O3/AVX2)",
        "prof_debug": "2. Debug (Символы отладки)",
        "select_extras": "Выберите дополнительные Python-пакеты:",
        "extra_viz": "1. Визуализация (matplotlib)",
        "extra_dash": "2. Веб-Дашборды (streamlit, plotly)",
        "extra_all": "3. Полная экосистема (всё вышеперечисленное + jupyter)",
        "extra_none": "4. Без доп. пакетов",
        "start_build": "🚀 Запуск процесса компиляции и установки...",
        "build_success": "🎉 TAAS Engine успешно собран и установлен!",
        "build_fail": "❌ Ошибка при сборке!",
        "diag_help": "💡 Советов по устранению:",
        "help_rust": "Установите Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
        "help_cuda": "Для включения CUDA убедитесь, что nvcc доступен в PATH",
    },
    "EN": {
        "title": "⚡ TAAS Engine — Interactive Setup & Build Wizard ⚡",
        "select_lang": "Select language / Выберите язык:",
        "sys_check": "🔍 System Environment Diagnostics",
        "tool": "Tool",
        "status": "Status",
        "version": "Version / Details",
        "found": "[bold green]Found[/bold green]",
        "missing": "[bold red]Missing[/bold red]",
        "optional_missing": "[bold yellow]Optional (Missing)[/bold yellow]",
        "comp_config": "⚙️ Configure Rust & CUDA Build Parameters",
        "select_features": "Select build feature set:",
        "opt_default": "1. Standard (Core + SIMD + Connectome + T3)",
        "opt_cuda": "2. GPU Acceleration (+ CUDA C++ Kernels)",
        "opt_minimal": "3. Minimal (Core SIMD only)",
        "opt_all": "4. Complete (All features + Python + CUDA)",
        "build_profile": "Select build profile:",
        "prof_release": "1. Release (Maximum O3/AVX2 optimization)",
        "prof_debug": "2. Debug (With debug symbols)",
        "select_extras": "Select optional Python extras:",
        "extra_viz": "1. Visualization (matplotlib)",
        "extra_dash": "2. Dashboards (streamlit, plotly)",
        "extra_all": "3. Full ecosystem (all extras + jupyter)",
        "extra_none": "4. No extra packages",
        "start_build": "🚀 Starting compilation and installation process...",
        "build_success": "🎉 TAAS Engine successfully built and installed!",
        "build_fail": "❌ Compilation / Installation Failed!",
        "diag_help": "💡 Troubleshooting advice:",
        "help_rust": "Install Rust toolchain: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
        "help_cuda": "For CUDA support, ensure `nvcc` is available in system PATH",
    }
}


def check_tool(cmd, version_flag="--version"):
    try:
        res = subprocess.run([cmd, version_flag], capture_output=True, text=True, check=True)
        out = res.stdout.splitlines()[0] if res.stdout else "OK"
        return True, out
    except Exception:
        return False, "Not found"


def run_tui():
    console.print(Panel("[bold cyan]TAAS Checkmate Neural Engine Setup[/bold cyan]", expand=False))
    
    # Language Selection
    lang_choice = Prompt.ask("Language / Язык", choices=["1", "2"], default="1")
    lang = "RU" if lang_choice == "1" else "EN"
    T = TEXTS[lang]

    console.print(Panel(f"[bold gold1]{T['title']}[/bold gold1]"))

    # System Diagnostics Table
    console.print(f"\n[bold]{T['sys_check']}[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column(T["tool"], style="dim")
    table.add_column(T["status"])
    table.add_column(T["version"])

    tools = [
        ("Python 3", sys.executable, "--version", True),
        ("Rustc (Rust compiler)", "rustc", "--version", True),
        ("Cargo", "cargo", "--version", True),
        ("Maturin (PyO3 builder)", "maturin", "--version", True),
        ("NVCC (CUDA compiler)", "nvcc", "--version", False),
    ]

    sys_ok = True
    cuda_found = False

    for name, cmd, flag, required in tools:
        found, ver = check_tool(cmd, flag)
        if name.startswith("NVCC") and found:
            cuda_found = True
        status = T["found"] if found else (T["missing"] if required else T["optional_missing"])
        if required and not found:
            sys_ok = False
        table.add_row(name, status, ver)

    console.print(table)

    if not sys_ok:
        console.print(f"\n[bold red]{T['build_fail']}[/bold red]")
        console.print(f"[yellow]{T['help_rust']}[/yellow]")
        return

    # Features Configuration
    console.print(f"\n[bold cyan]{T['comp_config']}[/bold cyan]")
    console.print(T["opt_default"])
    console.print(T["opt_cuda"])
    console.print(T["opt_minimal"])
    console.print(T["opt_all"])

    feat_choice = Prompt.ask(T["select_features"], choices=["1", "2", "3", "4"], default="1" if not cuda_found else "2")

    features_map = {
        "1": ["python", "connectome", "t3"],
        "2": ["python", "connectome", "t3", "cuda"],
        "3": ["python"],
        "4": ["all"],
    }
    selected_features = features_map[feat_choice]

    # Profile Choice
    prof_choice = Prompt.ask(T["build_profile"], choices=["1", "2"], default="1")
    release_mode = prof_choice == "1"

    # Extras Choice
    console.print(f"\n[bold cyan]Python Extras[/bold cyan]")
    console.print(T["extra_viz"])
    console.print(T["extra_dash"])
    console.print(T["extra_all"])
    console.print(T["extra_none"])
    extra_choice = Prompt.ask("Extras", choices=["1", "2", "3", "4"], default="1")

    extras_map = {
        "1": "viz",
        "2": "dashboard",
        "3": "all",
        "4": None,
    }
    selected_extra = extras_map[extra_choice]

    # Start Build Confirmation
    if not Confirm.ask(T["start_build"], default=True):
        return

    # Execute Build
    build_cmd = [
        "maturin", "develop",
        "--features", ",".join(selected_features)
    ]
    if release_mode:
        build_cmd.append("--release")

    env = os.environ.copy()
    env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Compiling Rust & CUDA extensions...", total=None)
        res = subprocess.run(build_cmd, cwd=root_dir, env=env, capture_output=True, text=True)

    if res.returncode == 0:
        console.print(Panel(f"[bold green]{T['build_success']}[/bold green]"))
        
        # Install Python Extras if selected
        if selected_extra:
            pip_cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{selected_extra}]"]
            subprocess.run(pip_cmd, cwd=root_dir)
            console.print("[green]Installed Python extras successfully![/green]")
    else:
        console.print(Panel(f"[bold red]{T['build_fail']}[/bold red]\n\n[dim]{res.stderr}[/dim]"))
        console.print(f"\n[yellow]{T['diag_help']}[/yellow]")
        if "nvcc" in res.stderr.lower() or "cuda" in res.stderr.lower():
            console.print(f"[yellow]{T['help_cuda']}[/yellow]")


if __name__ == "__main__":
    run_tui()
