from typing import List

def parse_input(raw: str) -> List[List[str]]:
    lines = raw.split("\n")
    chars = [list(line.strip()) for line in lines if line]

    assert chars  # not empty
    assert chars[0]  # not empty
    assert len({len(line) for line in chars}) == 1  # same lengths

    return chars


INPUT = """
EFGH
IJKL
MNOP
"""

CHARS = parse_input(INPUT)


def spiral(chars: List[List[str]], ilo=None, ihi=None, jlo=None, jhi=None) -> None:
    if ilo is None:
        ilo = jlo = 0
        ihi = len(chars) - 1
        jhi = len(chars[0]) - 1

    if ilo > ihi or jlo > jhi:
        return

    if ilo == ihi:
        for j in range(jlo, jhi+1):
            print(chars[ilo][j])
        return

    if jlo == jhi:
        for i in range(ilo, ihi+1):
            print(chars([i][jlo]))
        return

    # top ->
    for j in range(jlo, jhi):
        print(chars[ilo][j])

    # right down
    for i in range(ilo, ihi):
        print(chars[i][jhi])

    # <- bottom
    for j in range(jhi, jlo, -1):
        print(chars[ihi][j])

    # left up
    for i in range(ihi, ilo, -1):
        print(chars[i][jlo])

    # recurse
    spiral(chars, ilo + 1, ihi - 1, jlo + 1, jhi - 1)
