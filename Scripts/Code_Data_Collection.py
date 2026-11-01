import os
import io
import json
import random
import hashlib
import argparse
import subprocess

DEFAULT_EXTS = [".cc", ".h", ".c", ".cpp", ".hpp", ".hh", ".cxx", ".inl"]
DEFAULT_EXCLUDE_DIRS = [".git", ".github", "third-party", "third_party", "build", "docs",
                        "java", "buckifier", "coverage"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------
# Read Source
# ------------------------------------------------------------

def collect_rocksdb(root, exts, exclude_dirs, min_bytes, max_bytes):
    units, skipped = [], {"ext": 0, "size": 0, "non_utf8": 0}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in exclude_dirs)   # 排序：结果与文件系统的遍历顺序无关
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if os.path.splitext(fn)[1].lower() not in exts:
                skipped["ext"] += 1
                continue
            b = open(p, "rb").read()
            if not (min_bytes <= len(b) <= max_bytes):
                skipped["size"] += 1
                continue
            try:
                b.decode("utf-8")
            except UnicodeDecodeError:
                skipped["non_utf8"] += 1
                continue
            rel = os.path.relpath(p, root)
            units.append({"source": "rocksdb", "id": rel.replace(os.sep, "/"),
                          "dir": rel.split(os.sep)[0], "bytes": b})
    units.sort(key=lambda u: u["id"])
    return units, skipped


def load_records(name_or_path, hf_name, split):
    """优先本地文件（jsonl / json / parquet），否则用 HuggingFace datasets。"""
    if name_or_path and os.path.exists(name_or_path):
        if name_or_path.endswith(".parquet"):
            import pandas as pd
            return pd.read_parquet(name_or_path).to_dict("records")
        if name_or_path.endswith(".gz"):
            import gzip
            txt = gzip.open(name_or_path, "rt").read().strip()
        else:
            with open(name_or_path) as f:
                txt = f.read().strip()
        if txt.startswith("["):
            return json.loads(txt)
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
    from datasets import load_dataset
    ds = load_dataset(name_or_path or hf_name)
    recs = []
    for sp in (split if isinstance(split, list) else [split]):
        if sp in ds:
            recs += list(ds[sp])
    if not recs:                        # 有些版本只有一个 split
        for sp in ds:
            recs += list(ds[sp])
    return recs


def collect_mbpp(path):
    recs = load_records(path, "mbpp", ["train", "validation", "test", "prompt"])
    units, seen = [], set()
    for r in recs:
        tid = r.get("task_id")
        if tid in seen:
            continue
        seen.add(tid)
        code = (r.get("code") or "").strip()
        if code:
            units.append({"source": "mbpp", "id": f"mbpp/{tid}", "bytes": (code + "\n").encode()})
    return units


def collect_humaneval(path):
    recs = load_records(path, "openai_humaneval", ["test"])
    units = []
    for r in recs:
        code = (r.get("prompt") or "") + (r.get("canonical_solution") or "")
        if code.strip():
            units.append({"source": "humaneval", "id": r.get("task_id"), "bytes": code.encode()})
    return units


# ------------------------------------------------------------
# 划分
# ------------------------------------------------------------

def dedup(units):
    by_hash, out, dup = {}, [], 0
    for u in units:
        h = sha(u["bytes"])
        if h in by_hash:
            dup += 1
            continue
        by_hash[h] = u
        u["sha"] = h
        out.append(u)
    return out, dup


def split_rocksdb(units, test_bytes_needed, split_by, rng):
    if split_by == "dir":
        groups = {}
        for u in units:
            groups.setdefault(u["dir"], []).append(u)
        keys = sorted(groups)
        rng.shuffle(keys)
        test, train, acc = [], [], 0
        for k in keys:
            if acc < test_bytes_needed:
                test += groups[k]; acc += sum(len(u["bytes"]) for u in groups[k])
            else:
                train += groups[k]
        return train, test
    us = units[:]
    rng.shuffle(us)
    test, train, acc = [], [], 0
    for u in us:
        if acc < test_bytes_needed:
            test.append(u); acc += len(u["bytes"])
        else:
            train.append(u)
    return train, test


def split_by_frac(units, frac, rng):
    us = units[:]
    rng.shuffle(us)
    k = int(round(len(us) * frac))
    return us[k:], us[:k]


# ------------------------------------------------------------
# 生成样本
# ------------------------------------------------------------

def cut_at_newline(b, size):
    if len(b) <= size:
        return b
    i = b.rfind(b"\n", 0, size)
    return b[: (i + 1 if i > 0 else size)]


def build_test_samples(cpp_units, py_units, n, size, py_frac, rng):
    """每个样本：先放 py_frac 比例的 Python 片段，再用 C++ 文件填满；全部无放回。"""
    cpp, py = cpp_units[:], py_units[:]
    rng.shuffle(cpp); rng.shuffle(py)
    samples, used = [], []
    for i in range(n):
        buf, members = io.BytesIO(), []
        py_target = int(size * py_frac)
        while py and buf.tell() < py_target:
            u = py.pop(); buf.write(u["bytes"]); buf.write(b"\n"); members.append(u["id"])
        while cpp and buf.tell() < size:
            u = cpp.pop(); buf.write(u["bytes"]); buf.write(b"\n"); members.append(u["id"])
        if buf.tell() < size:
            raise RuntimeError(f"测试侧数据不够生成第 {i} 个样本，调大 --test_margin 或减小 --py_frac")
        samples.append(cut_at_newline(buf.getvalue(), size))
        used.append(members)
    return samples, used, cpp, py          # cpp/py 剩下的是没进任何测试样本的单元


def write_shards(units, out_dir, prefix, shard_size, rng):
    os.makedirs(out_dir, exist_ok=True)
    us = units[:]
    rng.shuffle(us)
    paths, buf, k = [], io.BytesIO(), 0

    def flush():
        nonlocal buf, k
        if buf.tell() == 0:
            return
        p = os.path.join(out_dir, f"{prefix}_{k:03d}")
        with open(p, "wb") as f:
            f.write(buf.getvalue())
        paths.append(p); k += 1; buf = io.BytesIO()

    for u in us:
        buf.write(u["bytes"]); buf.write(b"\n")
        if buf.tell() >= shard_size:
            flush()
    flush()
    return paths


# ------------------------------------------------------------
# From manifest Rebuilt（同一 RocksDB commit + MBPP → 逐字节相同的测试样本）
# ------------------------------------------------------------

def rebuild_from_manifest(args):
    man = json.load(open(args.from_manifest))
    margs = man["args"]
    exts, excl = set(margs["exts"]), set(margs["exclude_dirs"])
    try:
        commit = subprocess.check_output(["git", "-C", args.rocksdb_dir, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    if man.get("rocksdb_commit") and commit != man["rocksdb_commit"]:
        print(f"[Warn] RocksDB commit 不一致：manifest={man['rocksdb_commit']}，本地={commit}。"
              f"请先 git checkout {man['rocksdb_commit']}")

    pool = {}
    for u in collect_rocksdb(args.rocksdb_dir, exts, excl, margs["min_bytes"], margs["max_bytes"])[0]:
        pool[u["id"]] = u
    py_ids = [str(i) for i in man.get("python_train", []) + man.get("python_test", [])]
    py_ids += [str(i) for ms in man["test_samples"].values() for i in ms
               if str(i).startswith(("mbpp/", "HumanEval/"))]
    if any(i.startswith("mbpp/") for i in py_ids):
        for u in collect_mbpp(args.mbpp_path):
            pool[str(u["id"])] = u
    if any(i.startswith("HumanEval/") for i in py_ids):     # 论文的数据里含 HumanEval
        try:
            for u in collect_humaneval(args.humaneval_path):
                pool[str(u["id"])] = u
        except Exception as e:
            print(f"[Warn] 读取 HumanEval 失败（{type(e).__name__}）：manifest 需要 HumanEval，"
                  f"请用 --humaneval_path 指向 HumanEval.jsonl(.gz)")
    for d in args.extra_train_dirs:
        src = "extra:" + os.path.basename(os.path.normpath(d))
        for u in collect_rocksdb(d, exts, excl, margs["min_bytes"], margs["max_bytes"])[0]:
            pool[src + "/" + u["id"]] = u

    def get(uid):
        uid = uid.replace("\\", "/")
        return pool.get(uid)

    # 测试样本：按 manifest 记录的成员顺序拼接，再按原规则截断
    test_dir = os.path.join(args.out_dir, "test"); os.makedirs(test_dir, exist_ok=True)
    want = man.get("test_sample_sha256", {})
    ok_all, test_paths = True, []
    for name, members in man["test_samples"].items():
        buf, missing = io.BytesIO(), []
        for uid in members:
            u = get(str(uid))
            if u is None:
                missing.append(uid); continue
            buf.write(u["bytes"]); buf.write(b"\n")
        data = cut_at_newline(buf.getvalue(), margs["test_size"])
        p = os.path.join(test_dir, name); open(p, "wb").write(data); test_paths.append(p)
        h = sha(data)
        status = ("OK" if want.get(name) == h else "MISMATCH") if name in want else "no reference hash"
        ok_all &= (status != "MISMATCH") and not missing
        print(f"{name}: {len(data):,} bytes  sha256={h[:16]}…  {status}"
              + (f"  [缺少 {len(missing)} 个单元，例如 {missing[:3]}]" if missing else ""))

    # 训练分片：manifest 中列出的训练单元（顺序不影响结果的可比性）
    rng = random.Random(margs.get("seed", 0))
    groups = [("train_cpp", man.get("rocksdb_train", [])), ("train_extra", man.get("extra_train", [])),
              ("train_py", man.get("python_train", []))]
    train_paths = []
    for prefix, ids in groups:
        units = [get(str(i)) for i in ids]
        miss = sum(u is None for u in units)
        units = [u for u in units if u is not None]
        if miss:
            print(f"[Warn] {prefix}: manifest 中 {miss} 个单元本地找不到（extra 通常是 PyTorch 版本不同所致）")
        if units:
            train_paths += write_shards(units, os.path.join(args.out_dir, "train"), prefix, margs["shard_size"], rng)
    open(os.path.join(args.out_dir, "train_list.txt"), "w").write("\n".join(train_paths) + "\n")
    open(os.path.join(args.out_dir, "test_list.txt"), "w").write("\n".join(test_paths) + "\n")
    print(f"\n测试样本 {'全部与 manifest 一致' if ok_all else '存在不一致，见上'}；"
          f"训练分片 {len(train_paths)} 个 -> {os.path.join(args.out_dir, 'train')}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocksdb_dir", required=True)
    ap.add_argument("--mbpp_path", default="", help="本地 jsonl/json/parquet；留空则用 HF 'mbpp'")
    ap.add_argument("--humaneval_path", default="", help="本地文件；留空则用 HF 'openai_humaneval'")
    ap.add_argument("--no_python", action="store_true", help="不使用 MBPP / HumanEval")
    ap.add_argument("--no_humaneval", action="store_true", help="只用 MBPP，不用 HumanEval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_test", type=int, default=10)
    ap.add_argument("--test_size", type=int, default=1 << 20)
    ap.add_argument("--py_frac", type=float, default=0.05,
                    help="每个测试样本里 Python 片段的字节占比（MBPP+HumanEval 总量很小，默认 5%）")
    ap.add_argument("--py_test_frac", type=float, default=0.5,
                    help="Python 片段（按 task）划到 test 侧的比例")
    ap.add_argument("--test_margin", type=float, default=1.1, help="test 侧多留的余量倍数（未用到的部分会回收到训练集）")
    ap.add_argument("--split_by", choices=["file", "dir"], default="file")
    ap.add_argument("--extra_train_dirs", nargs="*", default=[],
                    help="额外的 C/C++ 源码目录，只用于训练（例如 LevelDB、PyTorch 的 C++ 头文件）")
    ap.add_argument("--exts", nargs="+", default=DEFAULT_EXTS)
    ap.add_argument("--exclude_dirs", nargs="+", default=DEFAULT_EXCLUDE_DIRS)
    ap.add_argument("--min_bytes", type=int, default=200)
    ap.add_argument("--max_bytes", type=int, default=512 * 1024)
    ap.add_argument("--shard_size", type=int, default=1 << 20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from_manifest", default="",
                    help="按已有 manifest.json 精确重建测试样本与训练分片（复现论文数据用）")
    args = ap.parse_args()
    if args.from_manifest:
        rebuild_from_manifest(args)
        return
    rng = random.Random(args.seed)

    # 1) 收集
    rdb, skipped = collect_rocksdb(args.rocksdb_dir, set(args.exts), set(args.exclude_dirs),
                                   args.min_bytes, args.max_bytes)
    py = []
    if not args.no_python:
        py = collect_mbpp(args.mbpp_path)
        if not args.no_humaneval:
            try:
                py += collect_humaneval(args.humaneval_path)
            except Exception as e:
                print(f"[Warn] 读取 HumanEval 失败（{type(e).__name__}），跳过 HumanEval。"
                      f"可用 --humaneval_path 指向本地 HumanEval.jsonl，或加 --no_humaneval")
    rdb, d1 = dedup(rdb)
    py, d2 = dedup(py)
    rdb_total = sum(len(u["bytes"]) for u in rdb)
    py_total = sum(len(u["bytes"]) for u in py)
    print(f"RocksDB: {len(rdb)} files, {rdb_total / 1e6:.1f} MB (去重 {d1}，跳过 {skipped})")
    print(f"Python : {len(py)} snippets, {py_total / 1e6:.2f} MB (去重 {d2})")

    # 2) 划分
    need_cpp = int(args.num_test * args.test_size * (1 - args.py_frac) * args.test_margin)
    rdb_train, rdb_test = split_rocksdb(rdb, need_cpp, args.split_by, rng)
    py_train, py_test = split_by_frac(py, args.py_test_frac, rng) if py else ([], [])
    need_py = args.num_test * args.test_size * args.py_frac
    if py and sum(len(u["bytes"]) for u in py_test) < need_py:
        print(f"[Warn] test 侧 Python 只有 {sum(len(u['bytes']) for u in py_test) / 1e3:.0f} KB，"
              f"不足以给每个样本 {args.py_frac:.0%}；样本里 Python 占比会偏低")

    # 3) 测试样本（只用 test 侧）
    test_dir = os.path.join(args.out_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    samples, members, cpp_left, py_left = build_test_samples(
        rdb_test, py_test, args.num_test, args.test_size, args.py_frac, rng)
    # 没进任何测试样本的 test 侧单元回收到训练侧（它们不在测试样本里，不构成泄漏）
    left_ids = {id(u) for u in cpp_left + py_left}
    rdb_test = [u for u in rdb_test if id(u) not in left_ids]
    py_test = [u for u in py_test if id(u) not in left_ids]
    rdb_train += cpp_left
    py_train += py_left
    print(f"回收未使用的 test 侧单元到训练集: {len(cpp_left)} 个 C++ 文件, {len(py_left)} 个 Python 片段, "
          f"{sum(len(u['bytes']) for u in cpp_left + py_left) / 1e6:.1f} MB")

    # 额外的训练专用 C/C++ 代码（只进训练侧；按内容哈希去掉与测试侧重复的）
    extra = []
    test_hashes = {u["sha"] for u in rdb_test + py_test}
    for d in args.extra_train_dirs:
        units, sk = collect_rocksdb(d, set(args.exts), set(args.exclude_dirs),
                                    args.min_bytes, args.max_bytes)
        for u in units:
            u["source"] = "extra:" + os.path.basename(os.path.normpath(d))
        units, _ = dedup(units)
        units = [u for u in units if u["sha"] not in test_hashes]
        print(f"额外训练代码 {d}: {len(units)} 个文件, {sum(len(u['bytes']) for u in units) / 1e6:.1f} MB")
        extra += units
    test_paths = []
    for i, s in enumerate(samples):
        p = os.path.join(test_dir, f"code_{i:02d}")
        open(p, "wb").write(s)
        test_paths.append(p)

    # 4) 训练分片（只用 train 侧）
    train_dir = os.path.join(args.out_dir, "train")
    train_paths = write_shards(rdb_train, train_dir, "train_cpp", args.shard_size, rng)
    if extra:
        train_paths += write_shards(extra, train_dir, "train_extra", args.shard_size, rng)
    if py_train:
        train_paths += write_shards(py_train, train_dir, "train_py", args.shard_size, rng)

    # 5) 自检：test 侧与 train 侧内容哈希不相交
    train_hashes = {u["sha"] for u in rdb_train + py_train + extra}
    assert not ({u["sha"] for u in rdb_test + py_test} & train_hashes), "train/test 内容重叠！"

    open(os.path.join(args.out_dir, "train_list.txt"), "w").write("\n".join(train_paths) + "\n")
    open(os.path.join(args.out_dir, "test_list.txt"), "w").write("\n".join(test_paths) + "\n")
    try:
        commit = subprocess.check_output(["git", "-C", args.rocksdb_dir, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    manifest = {
        "args": vars(args), "rocksdb_commit": commit,
        "rocksdb_train": sorted(u["id"] for u in rdb_train),
        "rocksdb_test": sorted(u["id"] for u in rdb_test),
        "python_train": sorted(str(u["id"]) for u in py_train),
        "python_test": sorted(str(u["id"]) for u in py_test),
        "extra_train": sorted(u["source"] + "/" + u["id"] for u in extra),
        "test_samples": {os.path.basename(p): m for p, m in zip(test_paths, members)},
        "test_sample_sha256": {os.path.basename(p): sha(open(p, "rb").read()) for p in test_paths},
        "stats": {
            "train_mb": sum(len(u["bytes"]) for u in rdb_train + py_train + extra) / 1e6,
            "extra_train_mb": sum(len(u["bytes"]) for u in extra) / 1e6,
            "test_pool_mb": sum(len(u["bytes"]) for u in rdb_test + py_test) / 1e6,
            "train_shards": len(train_paths),
        },
    }
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=1)
    print(f"\n测试样本: {len(test_paths)} 个 -> {test_dir}")
    print(f"训练分片: {len(train_paths)} 个, {manifest['stats']['train_mb']:.1f} MB -> {train_dir}")
    print(f"RocksDB commit: {commit}")
    print(f"清单: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
