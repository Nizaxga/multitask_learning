import ollama
import time

print("Generating embedding...")

x = ["This is a test sentence to check GPU usage."] * 1000
start = time.time()
embedding = ollama.embed(model="nomic-embed-text", input=x)
print("Done in", time.time() - start, "seconds")
