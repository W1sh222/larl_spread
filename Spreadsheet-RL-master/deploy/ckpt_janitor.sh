#!/bin/bash
# checkpoint 保留策略：永久保留 10 的倍数步 + 最新一份（续传用），其余删除。
# 用法: bash deploy/ckpt_janitor.sh <ckpt_dir>
DIR=${1:?用法: ckpt_janitor.sh <ckpt_dir>}
KEEP_EVERY=${KEEP_EVERY:-10}
# 额外永久保留的步数（逗号分隔），用于临时留档评估
KEEP_EXTRA=${KEEP_EXTRA:-}
while true; do
  latest=$(ls -d "$DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
  for d in "$DIR"/global_step_*; do
    [ -d "$d" ] || continue
    n=${d##*global_step_}
    [ "$n" = "$latest" ] && continue
    [ $((n % KEEP_EVERY)) -eq 0 ] && continue
    case ",$KEEP_EXTRA," in *",$n,"*) continue;; esac
    # 只删已写完(data.pt 存在)且 10 分钟无更新的目录，避免误删正在保存的
    if [ -f "$d/data.pt" ] && [ -z "$(find "$d" -maxdepth 0 -mmin -10 2>/dev/null)" ]; then
      rm -rf "$d" && echo "$(date '+%m-%d %H:%M') 删除 $d"
    fi
  done
  sleep 120
done
