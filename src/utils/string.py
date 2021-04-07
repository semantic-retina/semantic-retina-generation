BOLD_ENC = "\033[1m"
END_ENC = "\033[0m"


def bold(s: str):
    """Adds ANSI escape sequences to make a string bold when printed to the terminal."""
    return BOLD_ENC + s + END_ENC
