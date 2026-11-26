#!/usr/bin/env python3
"""从 test split 分层抽样构造可复现的验证集。

用法:
  # 生成 128 条验证集（默认 seed=42，与 data.seed 一致）
  python scripts/build_val_split.py --test data/test_hermes.parquet \
      --out data/val_128_seed42.parquet --n 128 --seed 42

  # 审计已有验证集：是否 test 子集、与训练集是否重叠、分层构成
  python scripts/build_val_split.py --check data/debug_val_64.parquet

分层键: answer_position 单/多区域 × prompt 长度三分位。
产出除 parquet 外，还会写一份 <out>.manifest.json，记录 seed、源文件 sha256、
被选中的 id 列表，保证任何一次抽样都能被复现和审计。
"""

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd


def prompt_hash(df: pd.DataFrame) -> pd.Series:
    return df["prompt"].astype(str).map(lambda s: hashlib.md5(s.encode()).hexdigest())


def strata_keys(df: pd.DataFrame, n_buckets: int = 3) -> pd.Series:
    """answer_position 单/多 × prompt 长度分位桶。"""
    pos = df["extra_info"].map(lambda e: "multi" if "," in str(e.get("answer_position")) else "single")
    length = df["prompt"].astype(str).str.len()
    # qcut 在长度分布重复值多时可能降级，fallback 到等宽分箱
    try:
        bucket = pd.qcut(length, n_buckets, labels=False, duplicates="drop")
    except ValueError:
        bucket = pd.cut(length, n_buckets, labels=False)
    return pos.astype(str) + "|len" + bucket.astype(str)


def build(test_df: pd.DataFrame, n: int, seed: int, n_buckets: int) -> pd.DataFrame:
    df = test_df.reset_index(drop=True).copy()
    df["_stratum"] = strata_keys(df, n_buckets)
    rng = np.random.RandomState(seed)

    counts = df["_stratum"].value_counts()
    # 按比例分配，余数按各层存量从大到小补
    quota = (counts / counts.sum() * n).astype(int).clip(lower=1)
    quota = quota.clip(upper=counts)
    short = n - int(quota.sum())
    if short > 0:
        pool = (counts - quota).sort_values(ascending=False)
        for k in pool.index:
            if short <= 0:
                break
            add = min(int(pool[k]), short)
            quota[k] += add
            short -= add

    picked = []
    for key, k in quota.items():
        idx = df.index[df["_stratum"] == key].to_numpy()
        picked.extend(rng.choice(idx, size=int(k), replace=False).tolist())

    out = df.loc[sorted(picked)].drop(columns=["_stratum"]).reset_index(drop=True)
    return out


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(df: pd.DataFrame, name: str) -> None:
    pos = df["extra_info"].map(lambda e: "multi" if "," in str(e.get("answer_position")) else "single")
    print(f"[{name}] {len(df)} 条 | single={int((pos == 'single').sum())} multi={int((pos == 'multi').sum())}"
          f" | prompt 长度 中位={int(df['prompt'].astype(str).str.len().median())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", help="test split parquet")
    ap.add_argument("--train", help="train split parquet，用于校验不重叠（可选）")
    ap.add_argument("--out", help="输出 parquet")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buckets", type=int, default=3, help="prompt 长度分位桶数")
    ap.add_argument("--check", help="只审计已有验证集文件，不生成")
    args = ap.parse_args()

    if args.check:
        val = pd.read_parquet(args.check)
        describe(val, "val")
        test_path = args.test or "data/test_hermes.parquet"
        test_df = pd.read_parquet(test_path)
        tv, tt = set(prompt_hash(val)), set(prompt_hash(test_df))
        print(f"val 是否为 test 子集: {'是' if tv <= tt else '否'}（{len(tv - tt)} 条不在 test 中）")
        if args.train:
            tr = pd.read_parquet(args.train)
            print(f"与 train 重叠: {len(tv & set(prompt_hash(tr)))} 条")
        try:
            mpath = args.check + ".manifest.json"
            with open(mpath, encoding="utf-8") as f:
                print(f"manifest 存在: {mpath} seed={json.load(f).get('seed')}")
        except OSError:
            print("manifest 不存在：该验证集无法追溯到抽样种子，不可复现")
        return 0

    if not (args.test and args.out):
        ap.error("生成模式需要 --test 和 --out")

    test_df = pd.read_parquet(args.test)
    val = build(test_df, args.n, args.seed, args.buckets)
    describe(test_df, "test")
    describe(val, "val")

    if args.train:
        tr = pd.read_parquet(args.train)
        overlap = set(prompt_hash(val)) & set(prompt_hash(tr))
        assert not overlap, f"与训练集重叠 {len(overlap)} 条"
        print("与训练集无重叠 ✓")

    val.to_parquet(args.out, index=False)
    manifest = {
        "seed": args.seed,
        "n": int(len(val)),
        "buckets": args.buckets,
        "source": args.test,
        "source_sha256": sha256_of(args.test),
        "strata": "answer_position(single/multi) x prompt_len_quantile",
        "ids": [str(e.get("id")) for e in val["extra_info"]],
    }
    with open(args.out + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"已写出 {args.out}（{len(val)} 条）与 {args.out}.manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
