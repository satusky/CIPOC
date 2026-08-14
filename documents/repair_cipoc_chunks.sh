#!/usr/bin/env bash
set -euo pipefail

chunk_dir="${1:-.}"
shopt -s nullglob
files=("$chunk_dir"/*_part_*.json)

if ((${#files[@]} == 0)); then
  printf 'No chunk files found in %s\n' "$chunk_dir" >&2
  exit 1
fi

for file in "${files[@]}"; do
  backup="${file}.bak"
  repaired="${file}.repaired"
  temporary="${file}.tmp"

  cp -p -- "$file" "$backup"

  # Guacamole may truncate Unicode punctuation or convert it through CP1252.
  LC_ALL=C sed \
    -e $'s/\xE2\x80\x99/\\\\u2019/g' \
    -e $'s/\xE2\x80\x9C/\\\\u201c/g' \
    -e $'s/\xE2\x80\x9D/\\\\u201d/g' \
    -e $'s/\xC2\xA0/\\\\u00a0/g' \
    -e $'s/\xC2\x92/\\\\u2019/g' \
    -e $'s/\xC2\x93/\\\\u201c/g' \
    -e $'s/\xC2\x94/\\\\u201d/g' \
    -e $'s/\x19/\\\\u2019/g' \
    -e $'s/\x1C/\\\\u201c/g' \
    -e $'s/\x1D/\\\\u201d/g' \
    -e $'s/\x92/\\\\u2019/g' \
    -e $'s/\x93/\\\\u201c/g' \
    -e $'s/\x94/\\\\u201d/g' \
    -e $'s/\xA0/\\\\u00a0/g' \
    "$file" > "$repaired"

  if jq --ascii-output . "$repaired" > "$temporary"; then
    mv -- "$temporary" "$file"
    rm -- "$repaired"
  else
    rm -f -- "$repaired" "$temporary"
    printf 'Could not repair %s; original and backup were preserved.\n' "$file" >&2
    exit 1
  fi
done

printf 'Repaired %d files. Backups use the .bak suffix.\n' "${#files[@]}"
