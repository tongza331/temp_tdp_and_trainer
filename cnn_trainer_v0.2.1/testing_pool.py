from multiprocessing import Pool
import time

def f(x):
    data = {
        "data1":{
            "class": "high_low",
            "confidence": "0.9"
        },
        "data2":{
            "class": "high_low",
            "confidence": "0.9"
        },
        "data3":{
            "class": "high_low",
            "confidence": "0.9"
        },
        "data4":{
            "class": "high_low",
            "confidence": "0.9"
        }
    }
    time.sleep(3)
    result = data[f"data{x}"]
    return result

if __name__ == "__main__":
    res = []
    x = [1,2,3,4]
    start = time.time()
    for i in x:
        res.append(f(i))
    end = time.time()
    print(f"Loop time:{end-start}")
    start2 = time.time()
    pool = Pool(processes=2)
    p1 = pool.map_async(f, x)
    res.append(p1.get())
    pool.close()
    end2 = time.time()
    print(f"Pool time:{end2-start2}")