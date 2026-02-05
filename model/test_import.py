# Test if we can import preprocessing
try:
    from preprocessing import extract_frames
    print("✓ SUCCESS! preprocessing imported correctly")
    print(f"extract_frames function found: {extract_frames}")
except Exception as e:
    print(f"✗ ERROR: {e}")
