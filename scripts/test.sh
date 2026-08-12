#!/bin/bash
# Comprehensive test suite for the Kyutai TTS Service.
#
# Usage:
#   ./scripts/test.sh                 # test default ports 7861 7862
#   ./scripts/test.sh 7861            # test a single port
#   ./scripts/test.sh 7861 7862 9000  # test multiple ports
#
# Requires: curl, ffprobe (static build in ~/bin), bc
# See docs/TESTING.md for details.

export PATH="$HOME/bin:$PATH"

# Ports to test (default: bare-metal 7861 + docker 7862)
if [ $# -gt 0 ]; then
    PORTS=("$@")
else
    PORTS=(7861 7862)
fi

PASS=0; FAIL=0; TOTAL=0

test_tts() {
    local PORT=$1 NAME=$2 DESC=$3 PAYLOAD=$4 OUT=$5
    local CODE SIZE
    CODE=$(curl -s -o "$OUT" -w "%{http_code}" -X POST "http://localhost:$PORT/api/tts" \
        -H "Content-Type: application/json" -d "$PAYLOAD")
    SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    if [ "$CODE" = "200" ] && [ "$SIZE" -gt 1000 ]; then
        echo "  ✅ [$NAME] $DESC (HTTP $CODE, ${SIZE}B)"
        PASS=$((PASS+1))
    else
        echo "  ❌ [$NAME] $DESC (HTTP $CODE, ${SIZE}B)"
        FAIL=$((FAIL+1))
    fi
    TOTAL=$((TOTAL+1))
}

for PORT in "${PORTS[@]}"; do
    DEP="port $PORT"
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  TESTING $DEP"
    echo "════════════════════════════════════════════════"

    # 1. ZipEnhancer status endpoint
    echo "--- 1. ZipEnhancer status ---"
    STATUS=$(curl -s "http://localhost:$PORT/api/zipenhancer/status")
    echo "$STATUS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
assert d['zipenhancer_available']==True, 'available'
assert d['pipeline_loaded']==True, 'pipeline_loaded'
print('  ✅ status: available=%s pipeline_loaded=%s' % (d['zipenhancer_available'], d['pipeline_loaded']))
" 2>/dev/null && PASS=$((PASS+1)) || { echo "  ❌ status endpoint"; FAIL=$((FAIL+1)); }
    TOTAL=$((TOTAL+1))

    # 2. Basic TTS MP3
    echo "--- 2. Basic TTS (MP3) ---"
    test_tts $PORT "$DEP" "basic mp3" '{"text":"Hello world, this is a basic test.","voice_choice":"Happy"}' /tmp/t_${PORT}_1.mp3

    # 3. Basic TTS WAV
    echo "--- 3. Basic TTS (WAV) ---"
    test_tts $PORT "$DEP" "basic wav" '{"text":"This is a WAV format test.","voice_choice":"Happy","output_format":"wav"}' /tmp/t_${PORT}_2.wav

    # 4. Custom filename
    echo "--- 4. Custom filename ---"
    test_tts $PORT "$DEP" "custom filename" '{"text":"Custom filename test.","voice_choice":"Happy","filename":"my_custom_audio"}' /tmp/t_${PORT}_3.mp3

    # 5. SSML multi-voice with breaks
    echo "--- 5. SSML multi-voice ---"
    test_tts $PORT "$DEP" "ssml multi-voice" '{"text":"<speak><voice name=\"Happy\">Hello!</voice><break time=\"1s\"/><voice name=\"Sad\">Goodbye.</voice><break time=\"2s\"/><voice name=\"Angry\">Enough!</voice></speak>","voice_choice":"Happy"}' /tmp/t_${PORT}_4.mp3

    # 6. ZipEnhancer standard
    echo "--- 6. ZipEnhancer standard ---"
    test_tts $PORT "$DEP" "zipenhancer standard" '{"text":"ZipEnhancer standard quality test.","voice_choice":"Happy","apply_zipenhancer":true,"zipenhancer_quality":"standard"}' /tmp/t_${PORT}_5.mp3

    # 7. ZipEnhancer high (wav)
    echo "--- 7. ZipEnhancer high ---"
    test_tts $PORT "$DEP" "zipenhancer high" '{"text":"ZipEnhancer high quality test.","voice_choice":"Happy","apply_zipenhancer":true,"zipenhancer_quality":"high","output_format":"wav"}' /tmp/t_${PORT}_6.wav

    # 8. ZipEnhancer ultra
    echo "--- 8. ZipEnhancer ultra ---"
    test_tts $PORT "$DEP" "zipenhancer ultra" '{"text":"ZipEnhancer ultra quality test.","voice_choice":"Happy","apply_zipenhancer":true,"zipenhancer_quality":"ultra","zipenhancer_window_size":1.0}' /tmp/t_${PORT}_7.mp3

    # 9. Audio effects
    echo "--- 9. Audio effects ---"
    test_tts $PORT "$DEP" "effects (normalize+boost+fades)" '{"text":"Audio effects test with normalization and fades.","voice_choice":"Happy","normalize":true,"volume_boost":3.0,"fade_in":500,"fade_out":1000,"bitrate":"192k"}' /tmp/t_${PORT}_8.mp3

    # 10. Complete pipeline
    echo "--- 10. Complete pipeline ---"
    test_tts $PORT "$DEP" "complete pipeline" '{"text":"Complete enhancement pipeline test.","voice_choice":"Happy","output_format":"wav","apply_zipenhancer":true,"zipenhancer_quality":"high","normalize":true,"volume_boost":2.0,"fade_in":500,"fade_out":1000,"filename":"complete_pipeline"}' /tmp/t_${PORT}_9.wav

    # 11. Error case: invalid voice (should fall back to default, still 200)
    echo "--- 11. Invalid voice (fallback) ---"
    test_tts $PORT "$DEP" "invalid voice fallback" '{"text":"Invalid voice test.","voice_choice":"NotARealVoice"}' /tmp/t_${PORT}_10.mp3

    # 12. Error case: empty text
    echo "--- 12. Empty text ---"
    CODE=$(curl -s -o /tmp/t_${PORT}_11.mp3 -w "%{http_code}" -X POST "http://localhost:$PORT/api/tts" \
        -H "Content-Type: application/json" -d '{"text":""}')
    TOTAL=$((TOTAL+1))
    if [ "$CODE" = "200" ] || [ "$CODE" = "400" ]; then
        echo "  ✅ [$DEP] empty text handled (HTTP $CODE)"
        PASS=$((PASS+1))
    else
        echo "  ❌ [$DEP] empty text (HTTP $CODE)"
        FAIL=$((FAIL+1))
    fi

    # 13. Audio validation with ffprobe
    echo "--- 13. Audio validation ---"
    for f in /tmp/t_${PORT}_1.mp3 /tmp/t_${PORT}_2.wav /tmp/t_${PORT}_4.mp3 /tmp/t_${PORT}_6.wav /tmp/t_${PORT}_9.wav; do
        DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null)
        FMT=$(ffprobe -v error -show_entries format=format_name -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null)
        TOTAL=$((TOTAL+1))
        if [ -n "$DUR" ] && [ "$(echo "$DUR > 0.5" | bc 2>/dev/null)" = "1" ]; then
            echo "  ✅ [$DEP] $(basename $f): $FMT, ${DUR}s"
            PASS=$((PASS+1))
        else
            echo "  ❌ [$DEP] $(basename $f): invalid ($FMT, ${DUR}s)"
            FAIL=$((FAIL+1))
        fi
    done

    # 14. SSML pause correctness (should be >= 3s for 1s+2s breaks)
    echo "--- 14. SSML pause correctness ---"
    DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 /tmp/t_${PORT}_4.mp3 2>/dev/null)
    TOTAL=$((TOTAL+1))
    if [ -n "$DUR" ] && [ "$(echo "$DUR > 3.0" | bc 2>/dev/null)" = "1" ]; then
        echo "  ✅ [$DEP] SSML breaks correct (${DUR}s >= 3s of pauses+speech)"
        PASS=$((PASS+1))
    else
        echo "  ❌ [$DEP] SSML breaks too short (${DUR}s)"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "  RESULTS: $PASS passed, $FAIL failed, $TOTAL total"
echo "════════════════════════════════════════════════"
[ "$FAIL" = "0" ] && echo "🎉 ALL TESTS PASSED" || echo "⚠️  $FAIL TEST(S) FAILED"
exit $FAIL
