import sys
import dex

# setup model
dex.eval()



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python demo.py path/to/img")
        sys.exit()
    
    path = sys.argv[1]
    age, female, male = dex.estimate(path)
    print("predict image: {}".format(path))
    print("woman: {:.3f}, man: {:.3f}".format(female, male))
    print("age: {:.3f}".format(age))
