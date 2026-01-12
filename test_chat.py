from rag_engine import build_index
from chat_engine import answer_question

print("Building index...")
ok = build_index("machine learning in medical imaging")

if not ok:
    print("Index build failed.")
    exit()

print("\nAsk a question:\n")

q1 = "What methods are used in these papers?"
print("Q:", q1)
print("A:", answer_question(q1))

print("\n----------------------\n")

q2 = "What datasets are used?"
print("Q:", q2)
print("A:", answer_question(q2))
