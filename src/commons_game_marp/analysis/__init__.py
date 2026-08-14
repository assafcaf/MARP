"""Analysis and plotting commands.

Each module here exposes exactly two public functions:

    build_parser() -> argparse.ArgumentParser
    run(args) -> int

`commons_game_marp.cli` imports a module only when its subcommand is actually
invoked, so nothing in this package is imported at CLI startup. Keep it that
way -- these modules pull in matplotlib and numpy at import time.
"""
