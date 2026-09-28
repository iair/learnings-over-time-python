#!/usr/bin/env bash
# Uso: bash /home/claude/serie2/_spec/verificar.sh <carpeta_del_modulo>
# Verifica TODOS los .py (Marimo) y .xlsx de la carpeta. Debe terminar con "TODO OK".
set -u
DIR="$1"
OUT=$(mktemp -d)
FAIL=0
for f in "$DIR"/*.py; do
  [ -e "$f" ] || continue
  echo "== notebook: $f"
  ( cd "$DIR" && timeout 300 python3 "$(basename "$f")" > "$OUT/run.log" 2>&1 ) || { echo "   ❌ python falla"; tail -25 "$OUT/run.log"; FAIL=1; continue; }
  marimo check "$f" > "$OUT/check.log" 2>&1 || { echo "   ❌ marimo check"; cat "$OUT/check.log" | head -30; FAIL=1; }
  ( cd "$DIR" && timeout 300 marimo export html "$(basename "$f")" -o "$OUT/x.html" > "$OUT/exp.log" 2>&1 ) || { echo "   ❌ export html (celdas con error)"; tail -15 "$OUT/exp.log"; FAIL=1; }
  echo "   ✓ ejecuta, check y export OK"
done
for x in "$DIR"/*.xlsx; do
  [ -e "$x" ] || continue
  echo "== planilla: $x"
  python3 /home/claude/serie2/_spec/verificar_xlsx.py "$x" || FAIL=1
done
if [ $FAIL -eq 0 ]; then echo "TODO OK"; else echo "HAY ERRORES"; exit 1; fi
