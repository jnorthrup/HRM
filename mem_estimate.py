def estimate_mem(batch, seq, hidden, layers):
    # bytes per element (bfloat16)
    B = 2
    act = 4 * batch * seq * hidden * layers   # forward+backward+cache
    return act / (1024**3)  # GB

if __name__ == "__main__":
    print(estimate_mem(8, 512, 4096, 24))  # ≈ 4.4 GB