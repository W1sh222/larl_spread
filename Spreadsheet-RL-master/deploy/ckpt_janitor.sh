#!/bin/bash
# checkpoint 保留策略：永久保留 KEEP_EVERY 的倍数步 + 最近 KEEP_LAST 份（续传/回退用），其余删除。
# 用法: bash deploy/ckpt_janitor.sh <ckpt_dir>
#   KEEP_EVERY=20 KEEP_LAST=2 bash deploy/ckpt_janitor.sh /path/to/exp_dir
DIR=${1:?用法: ckpt_janitor.sh <ckpt_dir>}
KEEP_EVERY=${KEEP_EVERY:-20}
KEEP_LAST=${KEEP_LAST:-2}
# 额外永久保留的步数（逗号分隔），用于临时留档评估
KEEP_EXTRA=${KEEP_EXTRA:-}
while true; do
  keep_recent=$(ls -d "$DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -n "$KEEP_LAST")
  for d in "$DIR"/global_step_*; do
    [ -d "$d" ] || continue
    n=${d##*global_step_}
    case " $(echo $keep_recent) " in *" $n "*) continue;; esac
    [ $((n % KEEP_EVERY)) -eq 0 ] && continue
    case ",$KEEP_EXTRA," in *",$n,"*) continue;; esac
    # 只删已写完(data.pt 存在)且 10 分钟无更新的目录，避免误删正在保存的
    if [ -f "$d/data.pt" ] && [ -z "$(find "$d" -maxdepth 0 -mmin -10 2>/dev/null)" ]; then
      rm -rf "$d" && echo "$(date '+%m-%d %H:%M') 删除 $d"
    fi
  done
  sleep 120
done
