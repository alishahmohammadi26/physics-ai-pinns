#!/usr/bin/env python3
"""
extract_linkedin_gif.py

Extracts the first (largest) animated GIF from the PINN Review blog post.
Saves it as linkedin_animation.gif for use with the LinkedIn post.

Usage:
    python3 extract_linkedin_gif.py
"""
import re, base64, os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../../../alishahmohammadi22.github.io/blog/pinn-review-2019-2026.html')

with open(SRC, 'r', encoding='utf-8') as f:
    content = f.read()

# Find all base64 GIFs
gifs = re.findall(r'data:image/gif;base64,([A-Za-z0-9+/=]+)', content)
print(f"Found {len(gifs)} GIFs in the article.")

if not gifs:
    print("No GIFs found.")
    exit(1)

# Pick the largest one (most likely the most detailed animation)
gifs_decoded = [(len(g), g) for g in gifs]
gifs_decoded.sort(reverse=True)

out_path = os.path.join(os.path.dirname(__file__), 'linkedin_animation.gif')
with open(out_path, 'wb') as f:
    f.write(base64.b64decode(gifs_decoded[0][1]))

size_kb = os.path.getsize(out_path) / 1024
print(f"Saved largest GIF ({size_kb:.0f} KB) to: {out_path}")
print("Use this file with the LinkedIn post infographic.")
print()
print("All GIF sizes (KB):")
for i, (sz, _) in enumerate(gifs_decoded[:10]):
    print(f"  GIF {i+1}: {sz*3//4//1024} KB (approx)")
