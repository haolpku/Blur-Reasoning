from model import get_model

print("Loading model...")
model = get_model('qwen2.5-vl-3b')
print("✓ Model loaded!")

print("\nTesting model call...")
response = model.call("1+1等于几？", history=[], system=None)
print(f"Response: {response}")
print("\n✓ Test successful!")
