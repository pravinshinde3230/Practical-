class item:
    def __init__(self,value,weight):
        self.value=value
        self.weight=weight

def fractionalKnapsack(w,arr):
    arr.sort(key=lambda x:(x.value/x.weight),reverse=True)
    finalvalue=0.0
    for item in arr:
        if item.weight <=w:
            w -=item.weight
            finalvalue +=item.value
        
        else:
            finalvalue +=item.value * w/item.weight
            break
    return finalvalue

if __name__ =="__main__":
    w=50
    arr=[item(60, 10),item(100, 20),item(120, 30)]
    max_val=fractionalKnapsack(w, arr)
    print('Maximum value we can obtain ={}'.format(max_val))