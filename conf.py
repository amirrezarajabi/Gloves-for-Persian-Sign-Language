CLASSES = [
    "Ast",
    "Baba",
    "Bad",
    "Bakht",
    "Khosh",
    "Mobark",
    "Mohandes",
]

CLASSES2INDEX = {k:v for v,k in enumerate(CLASSES)}

INDEX2CLASSES = {v:k for v,k in enumerate(CLASSES)}

def output_ctc(output):
    ans = ""
    tmp = "" if output[0] == "_" else output[0]
    ans = ans + tmp
    for i in range(len(output)):
        if output[i] != tmp and output[i] != "_":
            tmp = output[i]
            ans = ans + tmp
        if output[i] == "_":
            tmp = output[i]
    return ans

RANDOM_SEED = 42
