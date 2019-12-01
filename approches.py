l1=l1/255.0
l1=l1.astype(float)
l2=l2/255.0
l2=l2.astype(float)
laplcn_w1=l1
loclcontrst_w1=l1
expsd_w1=l1
laplcn_w2=l2
loclcontrst_w2=l2
expsd_w2=l2
# creating thread 

def woninput1():
    laplcn_w1 = laplacian(l1).astype(float)
    loclcontrst_w1 = localcontrast(l1).astype(float)
    #slncy_w1 = saliency(input_image_1).astype(float)
    expsd_w1 = Exposedness(l1).astype(float)


def woninput2():
    laplcn_w2 = laplacian(l2).astype(float)
    loclcontrst_w2 = localcontrast(l2).astype(float)
    #slncy_w2 = saliency(input_image_2).astype(float)
    expsd_w2 = Exposedness(l2).astype(float) 

p1 = Process(target=woninput1)
p2 =Process(target=woninput2)
p1.start()
p2.start()
p1.join()
p2.join()


def woninput1():
    t1 = threading.Thread(target=laplacian, args=(l1,)) 
    t2 = threading.Thread(target=localcontrast, args=(l1,)) 
    t3= threading.Thread(target=Exposedness,args=(l1,))
    #slncy_w1 = saliency(input_image_1).astype(float)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()


def woninput2():
    t1 = threading.Thread(target=laplacian, args=(l2,)) 
    t2 = threading.Thread(target=localcontrast, args=(l2,)) 
    t3= threading.Thread(target=Exposedness,args=(l2,))
    #slncy_w1 = saliency(input_image_1).astype(float)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()

p1 = Process(target=woninput1)
p2 =Process(target=woninput2)
p1.start()
p2.start()
p1.join()
p2.join()








imgpaths = ["2.jpg", "3.jpg","4.jpg","2.jpg"]
for imgpath in imgpaths:
    processes = []
    process = Process(target=fusionProcess, args=(imgpath,))
    processes.append(process)
    # process are spawned by creating a process object
    process.start()
