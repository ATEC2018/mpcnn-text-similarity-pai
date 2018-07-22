# coding=utf8

def main():
    a=1
    b=2

    def add(c):
        return a+b+c

    rst=add(3)
    print(rst)


if __name__=='__main__':
    main()