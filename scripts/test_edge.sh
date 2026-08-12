#!/bin/bash
# Edge-case test suite for the Kyutai TTS Service.
#
# Usage:
#   ./scripts/test_edge.sh            # test default port 7861
#   ./scripts/test_edge.sh 7862       # test a specific port
#
# Covers: all voices, filename header, invalid format fallback,
# bitrate verification, ZipEnhancer window size bounds.
# See docs/TESTING.md for details.

export PATH="$HOME/bin:$PATH"
PASS=0; FAIL=0
PORT=${1:-7861}

echo "=== 1. All 23 voices ==="
for v in Happy Angry Awe Calm Confused Desire Disgusted Laughing Sarcastic Sleepy Sleepy2 Default Enunciated Fast Projected Whisper Angry2 Awe2 Desire2 Sarcastic3 Child Sarcastic2 Default2; do
    CODE=$(curl -s -o /tmp/voice_test.mp3 -w "%{http_code}" -X POST http://localhost:$PORT/api/tts -H "Content-Type: application/json" -d "{\"text\":\"Voice test.\",\"voice_choice\":\"$v\"}")
    SIZE=$(stat -c%s /tmp/voice_test.mp3 2>/dev/null || echo 0)
    if [ "$CODE" = "200" ] && [ "$SIZE" -gt 1000 ]; then PASS=$((PASS+1)); else echo "  ❌ voice $v (HTTP $CODE, ${SIZE}B)"; FAIL=$((FAIL+1)); fi
done
echo "  ✅ $PASS/23 voices OK"

echo "=== 2. Custom filename header ==="
FN=$(curl -s -D - -o /dev/null -X POST http://localhost:$PORT/api/tts -H "Content-Type: application/json" -d "{\"text\":\"Filename test.\",\"filename\":\"my_special_name\"}" | grep -i "content-disposition" | tr -d "\r")
echo "  $FN"
echo "$FN" | grep -q "my_special_name.mp3" && { echo "  ✅ filename header correct"; PASS=$((PASS+1)); } || { echo "  ❌ filename header wrong"; FAIL=$((FAIL+1)); }

echo "=== 3. Invalid output_format (should default to mp3) ==="
CT=$(curl -s -D - -o /dev/null -X POST http://localhost:$PORT/api/tts -H "Content-Type: application/json" -d "{\"text\":\"Format test.\",\"output_format\":\"ogg\"}" | grep -i content-type | tr -d "\r")
echo "  $CT"
echo "$CT" | grep -qi "audio/mpeg" && { echo "  ✅ defaults to mp3"; PASS=$((PASS+1)); } || { echo "  ❌ wrong content type"; FAIL=$((FAIL+1)); }

echo "=== 4. Bitrate verification (128k) ==="
curl -s -o /tmp/br_test.mp3 -X POST http://localhost:$PORT/api/tts -H "Content-Type: application/json" -d "{\"text\":\"Bitrate test with a longer sentence to measure the actual bitrate.\",\"bitrate\":\"128k\"}"
BR=$(ffprobe -v error -show_entries format=bit_rate -of default=noprint_wrappers=1:nokey=1 /tmp/br_test.mp3 2>/dev/null)
echo "  actual bitrate: $BR bps"
if [ -n "$BR" ] && [ "$BR" -lt 160000 ]; then echo "  ✅ bitrate ~128k applied"; PASS=$((PASS+1)); else echo "  ❌ bitrate not applied ($BR)"; FAIL=$((FAIL+1)); fi

echo "=== 5. ZipEnhancer window size bounds (5.0) ==="
CODE=$(curl -s -o /tmp/ws_test.wav -w "%{http_code}" -X POST http://localhost:$PORT/api/tts -H "Content-Type: application/json" -d "{\"text\":\"Window size test.\",\"apply_zipenhancer\":true,\"zipenhancer_window_size\":5.0,\"output_format\":\"wav\"}")
SIZE=$(stat -c%s /tmp/ws_test.wav 2>/dev/null || echo 0)
[ "$CODE" = "200" ] && [ "$SIZE" -gt 1000 ] && { echo "  ✅ window 5.0 OK"; PASS=$((PASS+1)); } || { echo "  ❌ window 5.0 (HTTP $CODE)"; FAIL=$((FAIL+1)); }

echo ""
echo "EDGE RESULTS: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ] && echo "🎉 ALL EDGE TESTS PASSED" || echo "⚠️ $FAIL FAILED"
exit $FAIL
