"""
Phase 1.1 - Python Basics
Covers: variables, data types, type conversions
"""

# --- INTEGERS ---
age = 25
year = 2024
negative = -10
print(f"Integer: age={age}, year={year}, negative={negative}")
print(f"Type: {type(age)}")

# --- FLOATS ---
pi = 3.14159
temperature = -7.5
print(f"Float: pi={pi}, temperature={temperature}")
print(f"Type: {type(pi)}")

# --- STRINGS ---
name = "African LLM"
language = 'Swahili'
multiline = """This is a
multiline string"""
print(f"String: name={name}, language={language}")
print(f"Multiline: {multiline}")
print(f"Type: {type(name)}")

# --- BOOLEANS ---
is_training = True
is_deployed = False
print(f"Boolean: is_training={is_training}, is_deployed={is_deployed}")
print(f"Type: {type(is_training)}")

# --- TYPE CONVERSIONS ---
# int → float
x = float(42)
print(f"int to float: {x}, type: {type(x)}")

# float → int (truncates decimal)
y = int(3.99)
print(f"float to int: {y}, type: {type(y)}")

# int → string
s = str(100)
print(f"int to string: '{s}', type: {type(s)}")

# string → int
n = int("256")
print(f"string to int: {n}, type: {type(n)}")

# string → float
f = float("3.14")
print(f"string to float: {f}, type: {type(f)}")

# int → bool (0 is False, anything else is True)
print(f"bool(0)={bool(0)}, bool(1)={bool(1)}, bool(-5)={bool(-5)}")

# --- MINI PROGRAMS ---

# 1. Calculate area of a circle
radius = 5.0
area = pi * radius ** 2
print(f"\n1. Circle area (r={radius}): {area:.2f}")

# 2. Celsius to Fahrenheit
celsius = 100.0
fahrenheit = (celsius * 9/5) + 32
print(f"2. {celsius}°C = {fahrenheit}°F")

# 3. String operations
sentence = "building an llm from scratch"
print(f"3. Original: '{sentence}'")
print(f"   Uppercase: '{sentence.upper()}'")
print(f"   Word count: {len(sentence.split())}")
print(f"   Contains 'llm': {'llm' in sentence}")

# 4. Integer arithmetic
a, b = 17, 5
print(f"4. {a} / {b} = {a/b:.2f}")
print(f"   {a} // {b} = {a//b} (floor division)")
print(f"   {a} % {b} = {a%b} (remainder)")
print(f"   {a} ** {b} = {a**b} (power)")

# 5. Boolean logic
has_data = True
has_model = False
print(f"5. has_data AND has_model: {has_data and has_model}")
print(f"   has_data OR has_model: {has_data or has_model}")
print(f"   NOT has_data: {not has_data}")

# 6. Type checking
values = [42, 3.14, "hello", True, None]
for v in values:
    print(f"6. {repr(v):10} -> type: {type(v).__name__}")

# 7. String formatting
model_name = "FNI-LLM"
accuracy = 0.9342
params = 1_000_000
print(f"7. Model: {model_name}, Accuracy: {accuracy:.1%}, Params: {params:,}")

# 8. Numeric limits & special values
import math
print(f"8. math.inf={math.inf}, math.nan={math.nan}, math.pi={math.pi:.5f}")

# 9. String to list of characters
word = "Swahili"
chars = list(word)
print(f"9. '{word}' as chars: {chars}")

# 10. Chained type conversion
raw = "  42.7  "
result = int(float(raw.strip()))
print(f"10. '  42.7  ' -> strip -> float -> int = {result}")


if __name__ == "__main__":
    print("\nDone: Python basics complete.")
