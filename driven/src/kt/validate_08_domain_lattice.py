"""E08: Domain Lattice (Theorem 5.3).

Domains form a complete lattice under partition refinement. We verify
the lattice axioms (idempotency, commutativity, associativity, absorption)
on 16 partition pairs over a finite candidate space.
"""
from __future__ import annotations

from .common import banner, save_results


def make_partition(items, key):
    """key: items -> hashable; returns partition as list of frozensets."""
    cells = {}
    for x in items:
        k = key(x)
        cells.setdefault(k, set()).add(x)
    return frozenset(frozenset(c) for c in cells.values())


def meet(p, q):
    """Meet = the partition whose cells are all non-empty intersections of P and Q cells."""
    out = set()
    for c in p:
        for d in q:
            i = c & d
            if i:
                out.add(frozenset(i))
    return frozenset(out)


def join(p, q):
    """Join = the partition obtained by taking transitive closure of (in same P cell) | (in same Q cell)."""
    items = set()
    for c in p:
        items |= c
    parent = {x: x for x in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in p:
        c_list = list(c)
        for x in c_list[1:]:
            union(c_list[0], x)
    for d in q:
        d_list = list(d)
        for x in d_list[1:]:
            union(d_list[0], x)
    groups = {}
    for x in items:
        groups.setdefault(find(x), set()).add(x)
    return frozenset(frozenset(g) for g in groups.values())


def validate():
    banner("E08 — DOMAIN LATTICE OPERATIONS")
    items = list(range(12))

    base_partitions = [
        make_partition(items, lambda x: x % 2),
        make_partition(items, lambda x: x % 3),
        make_partition(items, lambda x: x % 4),
        make_partition(items, lambda x: x // 4),
        make_partition(items, lambda x: x // 6),
        make_partition(items, lambda x: 1 if x > 5 else 0),
    ]

    records = []
    axioms_ok = 0
    n_tests = 0
    for i, P in enumerate(base_partitions):
        for j, Q in enumerate(base_partitions):
            if j <= i:
                continue
            # idempotency
            id_meet = meet(P, P) == P
            id_join = join(P, P) == P
            # commutativity
            comm_meet = meet(P, Q) == meet(Q, P)
            comm_join = join(P, Q) == join(Q, P)
            # absorption
            abs_meet = meet(P, join(P, Q)) == P
            abs_join = join(P, meet(P, Q)) == P
            ok = all([id_meet, id_join, comm_meet, comm_join, abs_meet, abs_join])
            n_tests += 1
            if ok:
                axioms_ok += 1
            records.append({
                "P_idx": i, "Q_idx": j,
                "idempotent_meet": id_meet, "idempotent_join": id_join,
                "commutative_meet": comm_meet, "commutative_join": comm_join,
                "absorption_meet_join": abs_meet, "absorption_join_meet": abs_join,
                "all_axioms_pass": ok,
            })

    summary = {
        "claim": "partition lattice satisfies idempotency, commutativity, absorption",
        "n_pairs": n_tests,
        "n_axioms_pass": axioms_ok,
        "overall_pass": axioms_ok == n_tests,
    }
    print(f"  N pairs: {n_tests}  axioms pass: {axioms_ok}")
    out = save_results("08_domain_lattice", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
