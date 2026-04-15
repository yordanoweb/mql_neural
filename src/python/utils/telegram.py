"""Telegram notification helper. Reads BOT_TOKEN and CHAT_ID from .env at project root."""

import os
import urllib.request
import urllib.parse

def _load_env() -> tuple[str, str]:
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    token, chat_id = '', ''
    try:
        with open(os.path.normpath(env_path)) as f:
            for line in f:
                line = line.strip()
                if line.startswith('BOT_TOKEN='):
                    token = line.split('=', 1)[1].strip()
                elif line.startswith('CHAT_ID='):
                    chat_id = line.split('=', 1)[1].strip()
    except FileNotFoundError:
        pass
    return token, chat_id


_TOKEN, _CHAT_ID = _load_env()


def notify(text: str) -> None:
    """Fire-and-forget Telegram message. Silently ignores errors."""
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        url  = f'https://api.telegram.org/bot{_TOKEN}/sendMessage'
        data = urllib.parse.urlencode({'chat_id': _CHAT_ID, 'text': text}).encode()
        urllib.request.urlopen(url, data=data, timeout=5)
    except Exception:
        pass
