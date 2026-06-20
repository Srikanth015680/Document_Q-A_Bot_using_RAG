


import argparse
import sys
import textwrap

from src.query import query_rag_pipeline
from src.config import TOP_K


# DISPLAY HELPERS


SEPARATOR = "─" * 60
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def print_banner() -> None:
    print(f"\n{CYAN}{'═' * 60}")
    print("    DOCUMENT Q&A BOT — RAG Pipeline")
    print(f"{'═' * 60}{RESET}")
    print("  Ask questions about your documents.")
    print("  Type 'help' for commands | 'quit' to exit\n")


def print_help() -> None:
    print(
        f"""
{BOLD}Available Commands:{RESET}

  help          Show this help message
  sources       Toggle display of raw source chunks
  top-k <n>     Change number of retrieved chunks
  quit / exit   Exit the bot
"""
    )


def format_answer(result: dict, show_sources: bool = False) -> str:
    lines = [
        f"\n{GREEN}{BOLD}Answer:{RESET}",
        textwrap.fill(result["answer"], width=70),
        f"\n{YELLOW}📌 Sources used:{RESET}",
    ]

    for i, citation in enumerate(result["citations"], 1):
        lines.append(f"  [{i}] {citation}")

    if show_sources:
        lines.append(f"\n{BOLD}Retrieved Chunks:{RESET}")

        for i, (chunk, meta) in enumerate(
            zip(result["raw_chunks"], result["sources"]),
            start=1,
        ):
            distance = meta.get("distance")

            score_str = (
                f" (distance: {distance})"
                if distance is not None
                else ""
            )

            lines.append(
                f"\n  — Chunk {i} "
                f"[{meta['source']}, p.{meta['page']}]"
                f"{score_str}"
            )

            preview = (
                chunk[:300] + "…"
                if len(chunk) > 300
                else chunk
            )

            lines.append(
                textwrap.fill(
                    preview,
                    width=68,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )

    return "\n".join(lines)



# MAIN LOOP


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Document Q&A Bot (RAG)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve per query (default: {TOP_K})",
    )

    args = parser.parse_args()

    k = args.top_k
    show_sources = False

    print_banner()

    print("  Loading vector database …", end="", flush=True)

    try:
        from src.query import _get_collection

        _get_collection()

        print(f" {GREEN}✓ Ready{RESET}\n")

    except RuntimeError as exc:
        print(f"\n{RED}[ERROR] {exc}{RESET}\n")
        sys.exit(1)

    while True:
        try:
            user_input = input(f"{CYAN}You:{RESET} ").strip()

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{YELLOW}Goodbye! {RESET}\n")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in {"quit", "exit", "q"}:
            print(f"\n{YELLOW}Goodbye! {RESET}\n")
            break

        if lower == "help":
            print_help()
            continue

        if lower == "sources":
            show_sources = not show_sources

            state = "ON" if show_sources else "OFF"

            print(
                f"  Source chunks display: {state}\n"
            )
            continue

        if lower.startswith("top-k "):
            try:
                k = int(lower.split()[1])

                print(
                    f"  Retrieving top-{k} chunks per query.\n"
                )

            except (IndexError, ValueError):
                print(
                    f"  {RED}Usage: top-k <integer>{RESET}\n"
                )

            continue

        print(f"\n{SEPARATOR}")
        print(
            "  🔍 Searching documents …",
            end="",
            flush=True,
        )

        try:
            result = query_rag_pipeline(
                user_input,
                k=k,
            )

            print(
                f"\r{' ' * 35}\r",
                end="",
            )

            print(
                format_answer(
                    result,
                    show_sources=show_sources,
                )
            )

        except Exception as exc:
            print(
                f"\n{RED}[ERROR] {exc}{RESET}"
            )

        print(f"{SEPARATOR}\n")


if __name__ == "__main__":
    main()

