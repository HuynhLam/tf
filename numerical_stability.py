"""
numerical computation have some troubles with
calculate very big and very small values
"""

def main():
    X = 1
    print("{0:10.3f}".format(X))
    for i in range(0, 1000000):
        X += 0.000001
    print("{0:10.3f}".format(X))
    X += -1
    print("{0:10.3f}".format(X))
    return

if __name__ == "__main__":
    main()
