import hashlib
import random
import time

_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def make_ulid() -> str:
    ts = int(time.time() * 1000)
    ts_part = ""
    for _ in range(10):
        ts_part = _ULID_CHARS[ts & 0x1F] + ts_part
        ts >>= 5
    rand_part = "".join(random.choices(_ULID_CHARS, k=16))
    return ts_part + rand_part


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
