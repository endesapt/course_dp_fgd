#!/usr/bin/env bash
# Download Alibaba PAI v2020 tables used by the coursework.
# Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020
# License: CC BY 4.0.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p raw
cd raw

BASE="https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces"

for name in pai_job_table pai_task_table pai_machine_spec; do
  if [ -f "$name.csv" ]; then
    echo "exists: raw/$name.csv"
    continue
  fi
  echo "download: $name.tar.gz"
  curl -L --fail -O "$BASE/$name.tar.gz"
  echo "extract: $name.tar.gz"
  tar xzf "$name.tar.gz"
done

cd ..
sha256sum -c SHA256SUMS
echo "PAI v2020 ready under data/trace-gpu-v2020/raw/"
