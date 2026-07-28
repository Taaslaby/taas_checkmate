#!/usr/bin/env python3
"""CLI interface for taas neural toolkit."""

import sys
import click

@click.group()
def cli():
    """⚡ TAAS Checkmate Neural Engine CLI Toolkit ⚡"""
    pass

@cli.command(name="setup")
def setup_cmd():
    """Run the interactive TUI setup wizard."""
    from taas.setup_tui import run_tui
    run_tui()

@cli.command(name="info")
def info_cmd():
    """Display information about the installed TAAS engine."""
    click.echo("⚡ TAAS Checkmate Neural Engine v0.1.0")
    click.echo("Hardware SIMD: AVX2/NEON portable f32x8")
    try:
        import taas_checkmate
        click.echo("Rust C-Extension: Installed")
    except ImportError:
        click.echo("Rust C-Extension: Not installed (run `taas setup`)")

if __name__ == "__main__":
    cli()
