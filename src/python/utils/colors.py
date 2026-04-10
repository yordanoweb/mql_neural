try:
    import colorama
    colorama.init(autoreset=True)
    COLORS_SUPPORTED = True
except ImportError:
    COLORS_SUPPORTED = False

class Colors:
    if COLORS_SUPPORTED:
        RESET   = '\033[0m'
        RED     = '\033[91m'
        GREEN   = '\033[92m'
        YELLOW  = '\033[93m'
        BLUE    = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN    = '\033[96m'
        WHITE   = '\033[97m'
    else:
        RESET = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""

def colorize(text, color):
    if COLORS_SUPPORTED:
        return f"{color}{text}{Colors.RESET}"
    return text

# Aliases for compatibility
Fore = Colors
class Style:
    RESET_ALL = ""
c = colorize
