from matplotlib import pyplot as plt

def plotImage(fileName, data):
    plt.imsave("./output_img/conv/" + fileName + ".png", data)

    
def plotMultipleImages(fileName, data):
    print("data shape", data.shape)

    j = 0
    for d in data:
        print("d shape", d.shape)
        numberOfChannels = d.shape[2]
        for i in range(0, numberOfChannels):
            imgData = d[:,:,i]
            plt.imsave("./output_img/conv/" + fileName + "_" + str(j) + "_" + str(i) + ".png", imgData)
        j = j + 1

        if j == 2:
            break


# a = numpy.array([
#         numpy.array([
#             numpy.array([
#                 1,2
#             ]),
#             numpy.array([
#                 3,4
#             ]),
#         ]),
#         numpy.array([
#             numpy.array([
#                 5,6
#             ]),
#             numpy.array([
#                 7,8
#             ]),
        
#         ]),
#     ])