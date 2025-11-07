## clear up the file first
with open("agent_submission.py", "w") as f:
    f.write("")

with open("agent_submission.py", "a") as f:
    with open("ciphertext/ciphertext_decoder.py", "r") as g:
        f.write(g.read())

    with open("image_classification/image_classification.py", "r") as k:
        f.write(k.read())

    with open("agent.py", "r") as e:
        f.write(e.read())
