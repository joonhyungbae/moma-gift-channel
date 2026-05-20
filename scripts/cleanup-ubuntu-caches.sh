#!/usr/bin/env bash
# Ubuntu 캐시/불필요 파일 정리 - 공간 확보용
# 필요에 따라 sudo 로 실행하세요.

set -e

echo "=== 1. APT 캐시 정리 (sudo 필요) ==="
sudo apt-get clean
sudo apt-get autoclean
sudo apt-get autoremove -y

echo ""
echo "=== 2. 사용자 캐시 (~/.cache) 용량 확인 ==="
du -sh ~/.cache 2>/dev/null || true

echo ""
echo "=== 3. pip 캐시 정리 ==="
pip cache purge 2>/dev/null || true
pip3 cache purge 2>/dev/null || true

echo ""
echo "=== 4. 시스템 로그 정리 (오래된 journal, sudo 필요) ==="
echo "현재 journal 용량:"
journalctl --disk-usage 2>/dev/null || true
# 7일 이상 된 로그만 남기기 (선택)
# sudo journalctl --vacuum-time=7d

echo ""
echo "=== 5. 썸네일 캐시 (~/.cache/thumbnails) ==="
rm -rf ~/.cache/thumbnails/* 2>/dev/null && echo "썸네일 캐시 삭제됨" || true

echo ""
echo "=== 6. Trash 비우기 ==="
rm -rf ~/.local/share/Trash/* 2>/dev/null && echo "휴지통 비움" || true

echo ""
echo "=== 정리 후 디스크 여유 ==="
df -h /
